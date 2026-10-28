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


def logged(output, records):
    entries = journaled(output)
    processes = [(entry["role"], (output / "transcripts" / f"{entry['process']}.jsonl").read_text().splitlines())
                 for entry in entries if entry["entry"] == "process"]
    sessions = [lines for role, lines in processes if role == "inference"]
    learner, = [lines for role, lines in processes if role == "learner"]
    ends = [index + 1 for index, line in enumerate(learner) if json.loads(line).get("stage") == "released"]
    groups = [learner[start:end] for start, end in zip([0, *ends], ends)]
    attempts = [entry["binding"] for entry in entries if entry["entry"] == "attempt"]
    dispatched = {entry["request"]: entry["binding"] for entry in entries if entry["entry"] == "dispatched"}
    lines = []
    for index, (session, group, published, binding) in enumerate(zip(sessions, groups, records, attempts)):
        delivery = [dispatched[index * len(SEEDS) + position] for position in (1, 0)]
        lines += [*session, *group, json.dumps({"phase": "published", "checkpoint": str(output / f"generation{index + 1}"), "policy": published["policy"],
                                                "learner": published["learner"], "publication": "reference", "binding": binding, "delivery": delivery}),
                  json.dumps({"phase": "cycle", "index": index, "sessions": 1, "seconds": 0.5})]
    log = output.with_name(output.name + "log.jsonl")
    log.write_text("\n".join([*lines, *learner[ends[-1]:]]) + "\n")
    return log


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

    def test_a_default_two_cycle_run_journals_its_declaration_scores_the_reference_and_repeats_exactly(self):
        runs = {}
        for mode in ("serial", "resident"):
            with self.subTest(mode=mode):
                records, runs[mode] = self.train("default" + mode, 2, **{"inference-mode": mode})
                self.assertEqual(journaled(runs[mode])[0]["entry"], "declaration")
                published = [value["policy"] for value in records if value.get("phase") == "published"]
                self.assertEqual(len({self.settings["policy"], *published}), 3)
                self.assertEqual([value is not None for value in references(runs[mode])], [False, False, True, True])
        _, repeated = self.train("repeated", 2, staleness=0)
        comparison = compared({"run": runs["serial"], **independent(self.root, runs["serial"] / "generation2", "defaultserial")},
                              {"run": repeated, **independent(self.root, repeated / "generation2", "repeated")})
        self.assertTrue(comparison["equal"])
        self.assertTrue(comparison["schedule_equal"])
        self.assertTrue(comparison["execution"]["declared_equal"])

    def test_a_training_log_of_a_run_compares_equal_to_the_run_and_a_changed_second_cycle_is_named(self):
        records, output = self.train("logged", 2, staleness=0)
        self.assertEqual(len({self.settings["policy"], *(value["policy"] for value in records)}), 3)
        tasks = self.root / "loggedtasks.json"
        tasks.write_text(workload(2))
        declared = {key: self.settings[key] for key in ("policy", "learner", "reference-digest", "tokenizer-digest", "base-digest", "assembly-digest",
                                                         "behavior-base-digest", "behavior-assembly-digest", "clip", "penalty", "delta", "rate", "beta1",
                                                         "beta2", "optimizer-epsilon", "decay", "checkpoint", "reference", "inference-mode", "learning-mode",
                                                         "publication")}
        log = {**declared, "tasks": tasks, "log": logged(output, records), "sessions": 1, "output": output, "exit-code": 0,
               **independent(self.root, output / "generation2", "logged")}
        comparison = compared(log, {"run": output, **independent(self.root, output / "generation2", "loggedrun")})
        self.assertTrue(comparison["equal"])
        self.assertTrue(comparison["schedule_equal"])
        self.assertTrue(comparison["final_equal"])
        self.assertEqual([(generation["equal"], generation["states"]["equal"], generation["gradients"]["equal"], generation["probabilities_equal"])
                          for generation in comparison["generations"]], [(True, True, True, True)] * 2)
        for artifact in comparison["left"]["artifacts"]:
            self.assertEqual(artifact["learner_engine"]["path_log_ratio"]["kind"], "finite")
            self.assertGreaterEqual(artifact["learner_engine"]["tokens"], artifact["learner_engine"]["identical"])
        cycles = json.loads(workload(2))
        cycles[1]["tasks"][0]["seed"] += 1
        changed = self.root / "changed"
        execute([CORE, "train", *flags({**self.settings, "output": changed, "staleness": 0})], self.root / "changed.jsonl", stdin=json.dumps(cycles))
        comparison = compared(log, {"run": changed, **independent(self.root, changed / "generation2", "changed")})
        self.assertFalse(comparison["equal"])
        first, second = comparison["generations"]
        self.assertTrue(first["equal"])
        self.assertFalse(second["equal"])
        self.assertFalse(second["states"]["equal"])
        self.assertFalse(second["gradients"]["equal"])

    def test_a_training_path_contract_reads_the_run_history(self):
        _, output = self.train("pathcontract", 1, staleness=0, steps=2, **{"reference-source": "learner"})
        final = output / "generation1"
        described = core.invoke(["policy", "inspect", "--checkpoint", self.settings["checkpoint"]], executable=CORE)
        adapter = described["adapter"]
        identities = {"digest": adapter, "tokenizer-digest": described["tokenizer"], "base-digest": described["base"],
                      "assembly-digest": described["assembly"]}
        logs = {}
        for seed, call in zip((SEEDS[0], SEEDS[0], SEEDS[1], SEEDS[1]), (201, 202, 203, 204)):
            logs[call] = self.root / f"path{call}.jsonl"
            execute([CORE, "infer", *flags({**identities, "prompt": PROMPT, "tokens": 4, "temperature": 0.8, "seed": seed,
                                            "call": call, "attempt": call, "instance": call, "python": sys.executable,
                                            "worker": ENTRY, "cache": self.root, "adapter": self.settings["checkpoint"] / "adapter.safetensors"})], logs[call])
        runs = self.root / "pathruns.json"
        rows = []
        for name, seed, reference, candidate in zip(("0", "1"), SEEDS, (201, 203), (202, 204)):
            arguments = []
            for side, call in (("reference", reference), ("candidate", candidate)):
                arguments += [f"--{side}-log", str(logs[call]), f"--{side}-exit-code", "0",
                              f"--{side}-call", str(call), f"--{side}-attempt", str(call), f"--{side}-instance", str(call),
                              f"--{side}-prompt", PROMPT, f"--{side}-tokens", "4", f"--{side}-temperature", "0.8",
                              f"--{side}-seed", str(seed)]
                arguments += [f"--{side}-digest", adapter, f"--{side}-tokenizer-digest", described["tokenizer"],
                              f"--{side}-base-digest", described["base"], f"--{side}-assembly-digest", described["assembly"]]
            rows.append([0, name, arguments])
        runs.write_text(json.dumps(rows))
        reliance = ["ExecutionAuthenticity", "SelectedBehavior", "OwnCacheExecution", "PathAuthenticity", "PathBehavior",
                    "RequirementsJustified", "ContractFrozen", "AcceptanceIsolation", "SelectionControl"]
        policy = {"format": "invar-policy-v1", "model": described["model"], "revision": described["revision"],
                  "adapter": adapter, "tokenizer": described["tokenizer"], "base": described["base"], "assembly": described["assembly"]}
        contract = self.root / "pathcontract.json"
        contract.write_text(json.dumps({
            "format": "invar-use-contract/v2", "purpose": "Fixture training-path comparison",
            "domain": {"name": "training-path-fixture", "provenance_hex": "00" * 32, "unit_definition": "one task per unit",
                       "inputs": [{"cohort": 0, "key": name, "unit": name, "prompt": PROMPT, "tokens": 4,
                                   "temperature": 0.8, "seed": seed, "parameters": {}} for name, seed in zip(("0", "1"), SEEDS)]},
            "measurement": None,
            "reference": policy,
            "candidate": policy,
            "maximum_context": 1024,
            "criterion": {"numerical": [{"relation": {"kind": "training-path-token-log-gap", "learner": "candidate",
                                                       "budget": {"numerator": 1, "denominator": 2}},
                                         "probe_steps": [], "rationale": "Fixture per-token training-inference bound"}],
                          "loss": None, "invariance": []},
            "protocols": {"freeze": "Fixture only", "isolation": "Fixture only", "selection": "Fixture only"},
            "reliance": [{"premise": kind, "authority": "Fixture authority", "basis_sha256": "00" * 32} for kind in reliance],
            "transfers": []}))
        history = {"history-run": output, "history-codec": Path(__file__).resolve().parents[3] / "entries" / "codec.py",
                   "history-python": sys.executable,
                   **{"history-" + key: value for key, value in independent(self.root, final, "pathcontract").items()}}
        decision = core.invoke(["use", "admit", "--contract", contract, "--runs", runs, *flags(history)], executable=CORE)
        self.assertEqual(decision["decision"]["status"], "admitted_under_declared_reliance")
        self.assertFalse(any("ImplementationMismatch" in reason for reason in decision["decision"].get("reasons", [])))

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
