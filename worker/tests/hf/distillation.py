import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import contextlib
import hashlib
import json
import shutil
import subprocess

import torch
from safetensors.torch import save_file

from worker import float32
from worker.hf.model import FORMAT
from worker.hf.score import scored
from worker.tests.hf.overlap import CHILD_SECONDS, CORE, PROMPT, SEEDS, execute, flags, inspected, journaled, prepared, transcribed
from worker.tests.hf.scoring import MODELS, SCORE_ENTRY, TEACHER, identities

TEMPERATURE = 0.8


def workload(cycles):
    tasks = [{"name": str(index), "prompt": PROMPT, "tokens": 4, "seed": seed, "temperature": TEMPERATURE}
             for index, seed in enumerate(SEEDS)]
    return json.dumps([{"tasks": tasks, "order": [0, 1], "delivery": [1, 0]} for _ in range(cycles)])


def crashed(output, lines):
    (output / "journal.jsonl").write_bytes(b"".join(lines))
    reserved = {entry["process"] for entry in map(json.loads, lines) if entry["entry"] == "process"}
    for path in (output / "transcripts").iterdir():
        if int(path.stem) not in reserved:
            path.unlink()
    for path in output.iterdir():
        if path.name.startswith("generation"):
            path.unlink()
        elif path.name.startswith("staging"):
            shutil.rmtree(path)


def snapshot(root):
    return {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@contextlib.contextmanager
def changed(changes):
    originals = {path: path.read_bytes() if path.exists() else None for path in changes}
    for path, change in changes.items():
        written = change(originals[path])
        if written is None:
            path.unlink()
        else:
            path.write_bytes(written)
    try:
        yield
    finally:
        for path, original in originals.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)


def interleaved(encoded):
    lines = encoded.splitlines(keepends=True)
    entries = [json.loads(line) for line in lines]

    def answered(start):
        reading = next(index for index in range(start, len(entries)) if entries[index]["entry"] == "reading")
        completed = next(index for index in range(reading, len(entries)) if entries[index]["entry"] == "event" and entries[index]["event"]["kind"] == "completed" and entries[index]["event"]["request"] == entries[reading]["request"])
        return reading, completed

    first, done = answered(0)
    _, finished = answered(done + 1)
    return b"".join(lines[:first] + lines[done + 1:finished + 1] + lines[first:done + 1] + lines[finished + 1:])


def moved(lines, line, reading):
    kept = [value for value in lines if json.loads(value) != json.loads(line)]
    after = next(index for index, value in enumerate(kept) if json.loads(value) == reading)
    return kept[:after + 1] + [line] + kept[after + 1:]


def rebound(value, old, new):
    if value == old:
        return new
    if isinstance(value, dict):
        return {key: rebound(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [rebound(item, old, new) for item in value]
    return value


def joined(trajectory, call, records):
    words = next(record["observation"]["log_probability_bits"] for record in records if record["stage"] == "score_result" and record["binding"] == call)
    encoded = json.dumps({"format": "invar-distilled-result-v1", "trajectory": trajectory, "teacher": {"call": call, "words": words}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def replacement(output, entries, reading):
    taught = next(entry for entry in entries if entry["entry"] == "teacher" and entry["binding"] == reading["binding"])
    number = 1 + max(entry["process"] for entry in entries if entry["entry"] == "process")
    bound = dict.fromkeys(("attempt", "call", "instance"), 1 + max(entry["binding"]["call"] for entry in entries if "binding" in entry))
    records = [rebound(json.loads(line), taught["binding"], bound) for line in (output / "transcripts" / f"{taught['process']}.jsonl").read_bytes().splitlines()]
    stored = next(entry["digest"] for entry in entries if entry["entry"] == "result" and entry["request"] == reading["request"])
    digest = joined(stored, bound, records)
    reserved = next(entry for entry in entries if entry["entry"] == "process" and entry["process"] == taught["process"])
    ended = [entry for entry in entries[:entries.index(reading)] if entry["entry"] == "exit" and entry["process"] == taught["process"]]
    added = [{**reserved, "process": number}, {**taught, "binding": bound, "process": number}, *({**entry, "process": number} for entry in ended), {**reading, "binding": bound, "digest": digest}]
    completed = next(index for index, entry in enumerate(entries) if entry["entry"] == "event" and entry["event"]["kind"] == "completed" and entry["event"]["request"] == reading["request"])

    def journal(encoded):
        lines = encoded.splitlines(keepends=True)
        return b"".join(lines[:completed + 1] + [json.dumps(entry).encode() + b"\n" for entry in added] + lines[completed + 1:])

    return {output / "journal.jsonl": journal,
            output / "transcripts" / f"{number}.jsonl": lambda _: b"".join(json.dumps(record).encode() + b"\n" for record in records)}


class DistillationTests(unittest.TestCase):
    mode = "process"

    @classmethod
    def setUpClass(cls):
        cls.root, settings = prepared("invar-distillation-")
        cls.teacher = MODELS[TEACHER]()
        identity = identities(cls.teacher)
        adapter = cls.root / "teacher.safetensors"
        save_file({}, adapter)
        config = cls.root / "teacher.json"
        config.write_text(json.dumps({"format": FORMAT, "model": TEACHER[0], "revision": TEACHER[1]}))
        cls.settings = {**{key: value for key, value in settings.items() if key != "delta"},
                        "algorithm": "opd", "teacher": SCORE_ENTRY, "teacher-adapter": adapter, "teacher-config": config,
                        "teacher-model": TEACHER[0], "teacher-revision": TEACHER[1], "teacher-digest": identity["adapter"],
                        "teacher-base-digest": identity["base"], "teacher-assembly-digest": identity["assembly"],
                        "teacher-temperature": TEMPERATURE, "teacher-mode": cls.mode}
        cls.output = cls.root / "distilled"
        cls.records = execute([CORE, "train", *flags({**cls.settings, "output": cls.output, "staleness": 0})],
                              cls.root / "distilled.jsonl", stdin=workload(2))

    def test_each_rollout_completes_after_its_teacher_reading_and_the_update_learns_from_those_words(self):
        output, records = self.output, self.records
        self.assertEqual([(value["phase"], value["update"]) for value in records], [("published", 0), ("published", 1)])
        entries = journaled(output)
        teachers = {entry["process"] for entry in entries if entry["entry"] == "process" and entry["role"] == "teacher"}
        self.assertEqual({entry["process"]: entry["status"] for entry in entries if entry["entry"] == "exit" and entry["process"] in teachers},
                         {number: 0 for number in teachers})
        positions = {}
        for position, entry in enumerate(entries):
            if entry["entry"] in ("result", "teacher", "reading"):
                positions[entry["request"], entry["entry"]] = position
            elif entry["entry"] == "event" and entry["event"]["kind"] == "completed":
                positions[entry["event"]["request"], "completed"] = position
        readings = {entry["request"]: entry for entry in entries if entry["entry"] == "reading"}
        completions = {entry["event"]["request"]: entry["event"]["digest"] for entry in entries
                       if entry["entry"] == "event" and entry["event"]["kind"] == "completed"}
        self.assertEqual(sorted(readings), [0, 1, 2, 3])
        self.assertEqual(len(teachers), 4 if self.mode == "process" else 1)
        self.assertEqual({entry["process"] for entry in entries if entry["entry"] == "teacher"}, teachers)
        for request, reading in readings.items():
            self.assertLess(positions[request, "result"], positions[request, "teacher"])
            self.assertLess(positions[request, "teacher"], positions[request, "reading"])
            self.assertLess(positions[request, "reading"], positions[request, "completed"])
            self.assertEqual(completions[request], reading["digest"])
        consumed = [value["request"] for value in transcribed(output, "learner") if value.get("stage") == "consumed" and "samples" in value["request"]]
        self.assertEqual([request["specification"] for request in consumed], ["opd-token-mean/v1"] * 2)
        samples = [sample for request in consumed for sample in request["samples"]]
        self.assertEqual(sorted(json.dumps(sample["teacher_call"], sort_keys=True) for sample in samples),
                         sorted(json.dumps(reading["binding"], sort_keys=True) for reading in readings.values()))
        for sample in samples:
            direct = scored(self.teacher, torch.tensor([sample["tokens"]]), sample["prompt_length"], TEMPERATURE, device="cpu")
            self.assertEqual(sample["teacher_bits"], [float32.word(value) for value in direct.tolist()])
            self.assertNotEqual(sample["teacher_bits"], sample["behavior_bits"])

    def refused(self, output, changes):
        with changed(changes):
            before = snapshot(output)
            outcome = subprocess.run([CORE, "train", "--resume", str(output)], capture_output=True, text=True, timeout=CHILD_SECONDS)
            self.assertNotEqual(outcome.returncode, 0)
            self.assertEqual(snapshot(output), before)
            return outcome.stderr

    def test_readings_journaled_in_another_order_than_the_teacher_scored_them_resume(self):
        with changed({self.output / "journal.jsonl": interleaved}):
            resumed = execute([CORE, "train", "--resume", self.output], self.root / "interleaved.jsonl")
        self.assertEqual(resumed, [{"phase": "resumed", "committed": [0, 1]}])

    def test_a_resident_teacher_closed_after_a_released_score_without_its_reading_resumes(self):
        if self.mode == "process":
            self.skipTest("a teacher process exits after each score, before its reading is journaled")
        output = self.root / "closed"
        journal = output / "journal.jsonl"
        execute([CORE, "train", *flags({**self.settings, "output": output, "staleness": 0})], self.root / "closed.jsonl", stdin=workload(1))
        entries = journaled(output)
        last = max(index for index, entry in enumerate(entries) if entry["entry"] == "teacher")
        ended = next(index for index, entry in enumerate(entries) if entry["entry"] == "exit" and entry["process"] == entries[last]["process"])
        lines = journal.read_bytes().splitlines(keepends=True)
        crashed(output, lines[:last + 1] + [lines[ended]])
        records = execute([CORE, "train", "--resume", output], self.root / "closedresumed.jsonl")
        self.assertEqual(records[0], {"phase": "resumed", "committed": []})
        self.assertEqual([(value["phase"], value["update"]) for value in records[1:]], [("published", 0)])

    def test_a_finished_run_resumes_by_replaying_every_teacher_transcript(self):
        entries = journaled(self.output)
        journal = self.output / "journal.jsonl"
        reading = next(entry for entry in entries if entry["entry"] == "reading")
        changed = self.refused(self.output, {journal: lambda encoded: encoded.replace(reading["digest"].encode(), b"0" * 64, 1)})
        self.assertIn("A teacher reading differs from the teacher's admitted score", changed)
        teacher = next(entry["process"] for entry in entries if entry["entry"] == "process" and entry["role"] == "teacher")
        transcript = self.output / "transcripts" / f"{teacher}.jsonl"
        lines = transcript.read_bytes().splitlines(keepends=True)
        index = next(position for position, line in enumerate(lines) if json.loads(line)["stage"] == "score_result")
        result = json.loads(lines[index])
        words = result["observation"]["log_probability_bits"]
        result["observation"]["log_probability_bits"] = [words[0] + 1, *words[1:]]
        self.refused(self.output, {transcript: lambda _: b"".join(lines[:index]) + json.dumps(result).encode() + b"\n" + b"".join(lines[index + 1:])})
        teachers = [entry["process"] for entry in entries if entry["entry"] == "process" and entry["role"] == "teacher"]
        exit_line = next(json.dumps(entry) for entry in entries if entry["entry"] == "exit" and entry["process"] == teachers[0])
        cases = {"A teacher call names a process that is not a running teacher": lambda lines: [json.dumps({**json.loads(line), "role": "inference"}) if json.loads(line).get("entry") == "process" and json.loads(line)["process"] == teachers[0] else line for line in lines],
                 "A completed request has no stored result with its digest": lambda lines: [line.replace(reading["digest"], "1" * 64) if '"completed"' in line else line for line in lines]}
        if self.mode == "process":
            cases["A teacher reading is journaled before its process exited cleanly"] = lambda lines: moved(lines, exit_line, reading)
        else:
            released = next(position for position, line in enumerate(lines) if json.loads(line)["stage"] == "released")
            acknowledgement = {**json.loads(lines[released]), "result_sha256": "2" * 64}
            outcome = self.refused(self.output, {transcript: lambda _: b"".join(lines[:released]) + json.dumps(acknowledgement).encode() + b"\n" + b"".join(lines[released + 1:])})
            self.assertIn("Resident acknowledgement differs from its owner or exact completed inventory", outcome)
        for expected, change in cases.items():
            with self.subTest(expected=expected):
                outcome = self.refused(self.output, {journal: lambda encoded: ("\n".join(change(encoded.decode().splitlines())) + "\n").encode()})
                self.assertIn(expected, outcome)
        call = next(entry for entry in entries if entry["entry"] == "teacher")
        cut = self.refused(self.output, {journal: lambda encoded: b"".join(encoded.splitlines(keepends=True)[:entries.index(call) + 1]),
                            self.output / "transcripts" / f"{call['process']}.jsonl": lambda _: None})
        self.assertIn("An entry names a process whose transcript is missing", cut)
        replaced = self.refused(self.output, replacement(self.output, entries, reading))
        self.assertIn("A rollout's teacher reading is journaled twice", replaced)
        resumed = execute([CORE, "train", "--resume", self.output], self.root / "resumed.jsonl")
        self.assertEqual(resumed, [{"phase": "resumed", "committed": [0, 1]}])
        self.assertEqual([entry["entry"] for entry in journaled(self.output)[len(entries):]], ["restart"])

    def test_a_teacher_call_journaled_before_a_restart_reads_only_the_rollout_it_was_journaled_for(self):
        if self.mode == "resident":
            self.skipTest("the forged journal needs a teacher process that exited before the restart, and a resident teacher runs until the end")
        output = self.root / "restarted"
        journal = output / "journal.jsonl"
        execute([CORE, "train", *flags({**self.settings, "output": output, "staleness": 0})], self.root / "restarted.jsonl", stdin=workload(1))
        entries = journaled(output)
        first = next(entry for entry in entries if entry["entry"] == "teacher")
        kept = 1 + next(index for index, entry in enumerate(entries) if entry["entry"] == "exit" and entry["process"] == first["process"])
        crashed(output, journal.read_bytes().splitlines(keepends=True)[:kept])
        execute([CORE, "train", "--resume", output], self.root / "restartedresumed.jsonl")
        entries = journaled(output)
        restart = next(index for index, entry in enumerate(entries) if entry["entry"] == "restart")
        reading = next(index for index, entry in enumerate(entries) if index > restart and entry["entry"] == "reading" and entry["request"] == first["request"])
        completed = next(index for index, entry in enumerate(entries) if index > restart and entry["entry"] == "event" and entry["event"]["kind"] == "completed" and entry["event"]["request"] == first["request"])
        taught = next(entry for entry in entries if entry["entry"] == "teacher" and entry["binding"] == entries[reading]["binding"])
        stored = next(entry["digest"] for entry in entries[restart:] if entry["entry"] == "result" and entry["request"] == first["request"])
        records = [rebound(json.loads(line), taught["binding"], first["binding"]) for line in (output / "transcripts" / f"{taught['process']}.jsonl").read_bytes().splitlines()]
        (output / "transcripts" / f"{first['process']}.jsonl").write_bytes(b"".join(json.dumps(record).encode() + b"\n" for record in records))
        digest = joined(stored, first["binding"], records)
        replaced = {reading: {**entries[reading], "binding": first["binding"], "digest": digest},
                    completed: {**entries[completed], "event": {**entries[completed]["event"], "digest": digest}}}
        dropped = {index for index, entry in enumerate(entries) if entry["entry"] in ("process", "teacher", "exit") and entry["process"] == taught["process"]}
        lines = journal.read_bytes().splitlines(keepends=True)
        crashed(output, [json.dumps(replaced[index]).encode() + b"\n" if index in replaced else lines[index] for index in range(completed + 1) if index not in dropped])
        self.assertIn("Score consumption differs from the complete declared call", self.refused(output, {}))

    def test_the_history_admits_the_run_with_the_teacher_reading_of_every_sample(self):
        history = inspected(self.root, self.output, 2, "distilled")
        self.assertEqual(len(history["artifacts"]), 2)
        readings = [(json.dumps(reading["call"], sort_keys=True), reading["words"])
                    for generation in history["training"]["generations"] for reading in generation["cohort"]["teacher_readings"]]
        consumed = [value["request"] for value in transcribed(self.output, "learner") if value.get("stage") == "consumed" and "samples" in value["request"]]
        self.assertEqual(readings, [(json.dumps(sample["teacher_call"], sort_keys=True), sample["teacher_bits"]) for request in consumed for sample in request["samples"]])


class ResidentDistillationTests(DistillationTests):
    mode = "resident"


if __name__ == "__main__":
    unittest.main()
