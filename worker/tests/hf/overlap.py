import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from worker import core
from worker.hf import codec

ENTRY = Path(__file__).with_name("fixture.py").resolve()
CORE = os.environ.get("INVAR_CORE", "invar")
CHILD_SECONDS = 180
PROMPT = "Compute the answer."
SEEDS = (1326, 41)
BOOTSTRAP_BINDING = 7
FINAL_BINDING = 1000


def flags(values):
    return [str(item) for key, value in values.items() for item in ("--" + key, value)]


def workload(cycles):
    tasks = [{"name": str(index), "group": "group", "prompt": PROMPT, "tokens": 4, "seed": seed,
              "temperature": 0.8, "answer": "#### 437"} for index, seed in enumerate(SEEDS)]
    return json.dumps([{"tasks": tasks, "order": [0, 1], "delivery": [1, 0]} for _ in range(cycles)])


def execute(command, output, *, stdin=None, cwd=None):
    completed = subprocess.run(list(map(str, command)), input=stdin, capture_output=True, text=True, timeout=CHILD_SECONDS, cwd=cwd)
    output.write_text(completed.stdout)
    output.with_suffix(".stderr").write_text(completed.stderr)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return [json.loads(line) for line in completed.stdout.splitlines()]


def journaled(output):
    return [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]


def transcribed(output, role):
    return [json.loads(line) for entry in journaled(output) if entry["entry"] == "process" and entry["role"] == role
            for line in (output / "transcripts" / f"{entry['process']}.jsonl").read_text().splitlines()]


def references(output):
    scored = {}
    for record in transcribed(output, "inference"):
        if record.get("stage") == "result":
            for value in [json.loads(item) for item in record["calls"]] if "calls" in record else [record]:
                scored[value["binding"]["call"]] = value["reference"]
    return [scored[call] for call in sorted(scored)]


def independent(root, checkpoint, name):
    described = core.invoke(["policy", "inspect", "--checkpoint", checkpoint], executable=CORE)
    final = {"digest": described["adapter"], "tokenizer-digest": described["tokenizer"], "base-digest": described["base"],
             "assembly-digest": described["assembly"], "prompt": PROMPT, "tokens": 4, "temperature": 0.8, "seed": SEEDS[0],
             **{name: FINAL_BINDING for name in ("call", "attempt", "instance")}}
    log = root / (name + "final.jsonl")
    execute([CORE, "infer", *flags({**{key: final[key] for key in ("prompt", "tokens", "temperature", "seed", "call", "attempt", "instance")},
                                    "python": sys.executable, "worker": ENTRY, "cache": root, "checkpoint": checkpoint})], log)
    return {"cuda-rng-vectors": 0, "initial-source": "provided", "profile-mode": "unreported",
            "final-log": log, "final-exit-code": 0, **{"final-" + key: value for key, value in final.items()}}


def released(result, session):
    if session.tensors or session.views:
        raise AssertionError("History codec scope was not released")
    return result


def inspected(root, output, generations, name, *, process=False):
    fields = {"run": output, **independent(root, output / f"generation{generations}", name)}
    if process:
        codec_entry = Path(__file__).resolve().parents[3] / "entries" / "codec.py"
        return core.invoke(["inspect", "history", "--python", sys.executable, "--codec", codec_entry, *flags(fields)], executable=CORE)
    session = codec.Session()
    history = core.exchange(["inspect", "history", "--codec-mode", "stdio", *flags(fields)], executable=CORE, handler=session.handle)
    return released(history, session)


def compared(left, right):
    fields = {**{"left-" + key: value for key, value in left.items()}, **{"right-" + key: value for key, value in right.items()}}
    session = codec.Session()
    comparison = core.exchange(["compare", "histories", "--codec-mode", "stdio", *flags(fields)], executable=CORE, handler=session.handle)
    return released(comparison, session)


def prepared(prefix):
    root = Path(tempfile.mkdtemp(prefix=prefix))
    initial = root / "initial"
    observed, = execute([sys.executable, "-B", ENTRY, "--cache", root, "--output", initial], root / "initial.jsonl")
    inference = observed["inference"]
    inputs = {"digest": observed["policy"], "tokenizer-digest": observed["tokenizer"],
              "base-digest": inference["base"], "assembly-digest": inference["assembly"],
              "prompt": PROMPT, "tokens": 4, "temperature": 0.8, "seed": SEEDS[0],
              **{name: BOOTSTRAP_BINDING for name in ("call", "attempt", "instance")}}
    log = root / "bootstrap.jsonl"
    execute([CORE, "infer", *flags({**inputs, "python": sys.executable, "worker": ENTRY, "cache": root,
             "adapter": initial / "adapter.safetensors"})], log)
    core.invoke(["policy", *flags({**inputs, "checkpoint": initial, "log": log, "exit-code": 0})], executable=CORE)
    settings = {"policy": observed["policy"], "learner": observed["learner"], "reference-digest": observed["policy"],
                **{name + "-digest": observed[name] for name in ("tokenizer", "base", "assembly")},
                **{"behavior-" + name + "-digest": inference[name] for name in ("base", "assembly")},
                "clip": 0.2, "penalty": 0, "delta": 0.0001, "rate": 0.0001, "beta1": 0.9, "beta2": 0.999,
                "optimizer-epsilon": 0.00000001, "decay": 0,
                "python": sys.executable, "inference-python": sys.executable, "inference": ENTRY, "learning": ENTRY,
                "inference-mode": "serial", "learning-mode": "resident", "cache": root, "checkpoint": initial,
                "reference": initial / "adapter.safetensors", "publication": "reference"}
    return root, settings


class OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.settings = prepared("invar-overlap-")

    def train(self, name, cycles, **extra):
        output = self.root / name
        records = execute([CORE, "train", *flags({**self.settings, "output": output, **extra})],
                          self.root / (name + ".jsonl"), stdin=workload(cycles))
        return records, output

    def test_the_next_rollout_runs_while_the_stale_update_learns(self):
        records, output = self.train("stale", 2, staleness=1)
        self.assertEqual([(value["phase"], value["update"]) for value in records], [("published", 0), ("published", 1)])
        self.assertNotEqual(records[0]["policy"], self.settings["policy"])
        consumed = [value["request"] for value in transcribed(output, "learner") if value.get("stage") == "consumed" and "samples" in value["request"]]
        self.assertEqual([request["schedule"] for request in consumed], [{"update": 0, "staleness": 1}, {"update": 1, "staleness": 1}])
        self.assertEqual({(sample["version"], sample["behavior_policy"]) for request in consumed for sample in request["samples"]},
                         {(0, self.settings["policy"])})
        self.assertEqual(consumed[1]["policy"], records[0]["policy"])
        entries = journaled(output)
        resumed = execute([CORE, "train", "--resume", output], self.root / "staleresumed.jsonl")
        self.assertEqual(resumed, [{"phase": "resumed", "committed": [0, 1]}])
        self.assertEqual([entry["entry"] for entry in journaled(output)[len(entries):]], ["restart"])
        reserved = [entry["process"] for entry in entries if entry["entry"] == "process"]
        self.assertEqual({entry["process"]: (entry["outcome"], entry["status"], entry["output"]) for entry in entries if entry["entry"] == "exit"},
                         {number: ("exited", 0, "complete") for number in reserved})
        self.assertEqual(sorted(path.name for path in (output / "transcripts").iterdir()), sorted(f"{number}.jsonl" for number in reserved))
        intervals = {(entry["role"], entry["update"]): (entry["start"], entry["end"]) for entry in entries if entry["entry"] == "interval"}
        self.assertEqual(sorted(intervals), [("learner", 0), ("learner", 1), ("rollout", 0), ("rollout", 1)])
        rollout, learner = intervals["rollout", 1], intervals["learner", 0]
        self.assertLess(rollout[0], learner[1])
        self.assertLess(learner[0], rollout[1])
        native = torch.load(output / "generation2" / "learner.pt", weights_only=True)
        self.assertTrue(all(slot["step"].item() == 2 for slot in native["optimizer"]["state"].values()))
        history = inspected(self.root, output, 2, "stale")
        self.assertEqual(len(history["artifacts"]), 2)
        self.assertEqual(history["training"]["staleness"], 1)
        self.assertEqual([attempt["outcome"] for attempt in history["training"]["attempts"]], [{"committed": 1}, {"committed": 2}])

    def test_two_optimizer_steps_per_update_are_counted_in_the_history(self):
        _, output = self.train("twosteps", 1, staleness=0, steps=2)
        history = inspected(self.root, output, 1, "twosteps")
        steps, = [item["state"]["steps"] for item in history["artifacts"]]
        self.assertTrue(steps and all(count == 2 for count in steps))

    def test_the_default_train_run_uses_the_event_runtime_at_staleness_zero(self):
        for mode in ("serial", "resident"):
            with self.subTest(mode=mode):
                records, output = self.train("default" + mode, 2, **{"inference-mode": mode})
                self.assertEqual([value["phase"] for value in records], ["published", "published"])
                published = [value["policy"] for value in records]
                self.assertEqual(len({self.settings["policy"], *published}), 3)
                self.assertEqual([value is not None for value in references(output)], [False, False, True, True])
                history = inspected(self.root, output, 2, "default" + mode)
                self.assertEqual(history["training"]["staleness"], 0)
                resumed = execute([CORE, "train", "--resume", output], self.root / ("default" + mode + "resumed.jsonl"))
                self.assertEqual(resumed, [{"phase": "resumed", "committed": [0, 1]}])

    def test_staleness_is_a_schedule_difference(self):
        _, fresh = self.train("fresh", 1, staleness=0)
        _, lagged = self.train("lagged", 1, staleness=1)
        comparison = compared({"run": fresh, **independent(self.root, fresh / "generation1", "fresh")},
                              {"run": lagged, **independent(self.root, lagged / "generation1", "lagged")})
        self.assertFalse(comparison["schedule_equal"])
        self.assertEqual(comparison["schedule"]["differences"], [{"generation": 1, "fields": ["staleness"]}])
        self.assertEqual([[update["staleness"] for update in comparison["schedule"][side]] for side in ("left", "right")], [[0], [1]])
        self.assertTrue(comparison["execution"]["declared_equal"])
        self.assertFalse(comparison["equal"])


if __name__ == "__main__":
    unittest.main()
