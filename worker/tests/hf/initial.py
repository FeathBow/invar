import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from worker.hf import codec
from worker import core
from worker.hf.probe import checkpoint, digest
from worker.tests.hf.learning import make_learner
from worker.tests.hf.states import sha
from worker.tests.hf.tokenization import make_tokenizer

SEED = 17


def arguments(path, policy, learner):
    return {"checkpoint": str(path), "policy": policy, "learner": sha(path / "learner.pt"),
            "reference-digest": policy, **{key + "-digest": learner[key] for key in ("tokenizer", "base", "assembly")},
            **{"behavior-" + key + "-digest": learner[key] for key in ("base", "assembly")},
            "clip": "0.2", "penalty": "0", "delta": "0.0001", "rate": "0.0001", "beta1": "0.9", "beta2": "0.999",
            "optimizer-epsilon": "0.00000001", "decay": "0", "cuda-rng-vectors": "0", "initial-source": "provided"}


def inspect(fields):
    session = codec.Session()
    command = ["inspect", "initial", "--codec-mode", "stdio", *[item for key, value in fields.items() for item in ("--" + key, value)]]
    value = core.exchange(command, executable="invar", handler=session.handle)
    if session.tensors or session.views:
        raise AssertionError("Initial checkpoint codec scope was not released")
    return value


class InitialTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="invar-initial-"))
        learner = make_learner()
        policy = checkpoint(learner.model, learner.optimizer, self.path, tokenizer=make_tokenizer(), expected=None)
        actual = torch.load(self.path / "learner.pt", weights_only=True)
        self.fields = arguments(self.path, digest(policy), actual)

    def test_actual_initial_snapshot_is_admitted_without_self_comparison(self):
        result = inspect(self.fields)
        self.assertEqual(result["source"], {"kind": "provided"})
        self.assertEqual(result["state"]["steps"], [])
        self.assertEqual(result["state"]["cuda_rng_bytes"], [])
        self.assertEqual(result["learner"], sha(self.path / "learner.pt"))

    def test_initial_snapshot_requires_its_actual_materialization_and_rng_inventory(self):
        for changes in ({"policy": "a" * 64}, {"learner": "a" * 64}, {"base-digest": "a" * 64},
                        {"rate": "0.5"}, {"cuda-rng-vectors": "1"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                inspect({**self.fields, **changes})

    def contract(self):
        measurement = {"seconds": 0, "peak_allocated": 0, "peak_reserved": 0}
        report = {"stage": "initial", "policy": self.fields["policy"], "learner": self.fields["learner"],
                  **{key: self.fields[key + "-digest"] for key in ("tokenizer", "base", "assembly")},
                  "seed": SEED, "optimizer_steps": 0, "scope": "report contract fixture; initializer not executed"}
        return [{"stage": "loading"}, {"stage": "profile", "model": "fixture", "revision": "fixture"},
                {"stage": "load", **measurement}, {"stage": "checkpoint", **measurement}, report]

    def reported(self, records, **changes):
        log = self.path / "initial.jsonl"
        log.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        return inspect({**self.fields, "initial-source": "initializer", "initial-seed": str(SEED),
                        "initial-exit-code": "0", "initial-log": str(log), **changes})

    def test_initializer_report_contract_binds_exit_seed_and_complete_stream(self):
        records = self.contract()
        result = self.reported(records)
        self.assertEqual(result["source"]["log_sha256"], sha(self.path / "initial.jsonl"))
        self.assertEqual(result["source"]["seed"], SEED)
        for changes in ({"initial-exit-code": "1"}, {"initial-seed": "18"}, {"initial-source": "provided"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.reported(records, **changes)
        for altered in (records[:-1], records + records[-1:], records[::-1]):
            with self.subTest(records=altered), self.assertRaises(ValueError):
                self.reported(altered)

    def test_initializer_report_contract_cannot_replace_actual_state_checks(self):
        for key, value in (("policy", "a" * 64), ("tokenizer", "a" * 64), ("optimizer_steps", 1)):
            records = copy.deepcopy(self.contract())
            records[-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.reported(records)
        path = self.path / "learner.pt"
        actual = torch.load(path, weights_only=True)
        actual["optimizer"]["state"] = {0: {}}
        torch.save(actual, path)
        changed = {**self.fields, "learner": sha(path)}
        with self.assertRaisesRegex(ValueError, "empty initial AdamW state"):
            inspect(changed)


if __name__ == "__main__":
    unittest.main()
