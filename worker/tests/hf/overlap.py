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

ENTRY = Path(__file__).with_name("fixture.py").resolve()
CORE = os.environ.get("INVAR_CORE", "invar")
CHILD_SECONDS = 180
PROMPT = "Compute the answer."
SEEDS = (1326, 41)
BOOTSTRAP_BINDING = 7


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
        publications = [value for value in records if value.get("phase") == "published"]
        self.assertEqual([value["update"] for value in publications], [0, 1])
        self.assertNotEqual(publications[0]["policy"], self.settings["policy"])
        consumed = [value["request"] for value in records if value.get("stage") == "consumed" and "samples" in value["request"]]
        self.assertEqual([request["schedule"] for request in consumed], [{"update": 0, "staleness": 1}, {"update": 1, "staleness": 1}])
        self.assertEqual({(sample["version"], sample["behavior_policy"]) for request in consumed for sample in request["samples"]},
                         {(0, self.settings["policy"])})
        self.assertEqual(consumed[1]["policy"], publications[0]["policy"])
        entries = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
        intervals = {(entry["role"], entry["update"]): (entry["start"], entry["end"]) for entry in entries if entry["entry"] == "interval"}
        self.assertEqual(sorted(intervals), [("learner", 0), ("learner", 1), ("rollout", 0), ("rollout", 1)])
        rollout, learner = intervals["rollout", 1], intervals["learner", 0]
        self.assertLess(rollout[0], learner[1])
        self.assertLess(learner[0], rollout[1])
        native = torch.load(output / "generation2" / "learner.pt", weights_only=True)
        self.assertTrue(all(slot["step"].item() == 2 for slot in native["optimizer"]["state"].values()))

    def test_zero_staleness_publishes_the_synchronous_successor(self):
        synchronous, left = self.train("synchronous", 1)
        concurrent, right = self.train("concurrent", 1, staleness=0)
        expected, = [value for value in synchronous if value.get("phase") == "published"]
        actual, = [value for value in concurrent if value.get("phase") == "published"]
        self.assertEqual({key: actual[key] for key in ("policy", "learner", "publication")},
                         {key: expected[key] for key in ("policy", "learner", "publication")})
        self.assertNotEqual(actual["policy"], self.settings["policy"])
        self.assertEqual((right / "generation1" / "policy.json").read_bytes(), (left / "generation1" / "policy.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
