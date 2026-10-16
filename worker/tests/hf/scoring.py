import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import save_file

from worker import float32
from worker.hf import assembly, frozen, operation, tensors
from worker.hf.inference import Runtime, installed
from worker.hf.model import FORMAT, adapter_state
from worker.hf.policy import read_adapter
from worker.hf.score import scored
from worker.implementation import INFERENCE
from worker.tests.hf.handshake import cpu_measure
from worker.tests.hf.inference import IDENTITY, TEST_THREADS, model
from worker.tests.hf.tokenization import make_tokenizer

CORE = os.environ.get("INVAR_CORE", "invar")
SOURCE_ENTRY = Path(__file__).with_name("fixture.py").resolve()
SCORE_ENTRY = Path(__file__).with_name("scorefixture.py").resolve()
CHILD_SECONDS = 180
TEACHER = ("local-test/Llama-teacher", "2" * 40)
TEACHER_SEED = 43
MODELS = {IDENTITY: model, TEACHER: partial(model, seed=TEACHER_SEED, adapted=False)}


def flags(values):
    return [str(item) for key, value in values.items() for item in ("--" + key, value)]


def load(cache, adapter, *, expected, emit, declared):
    torch.set_num_threads(TEST_THREADS)
    state = read_adapter(adapter, expected["adapter"])
    tokenizer = make_tokenizer()
    operation.verify(tokenizer, expected["tokenizer"])
    loaded = cpu_measure("load", MODELS[declared], emit=emit)
    installed(loaded, state, expected)
    return Runtime(model=loaded, tokenizer=tokenizer, adapter=adapter, device="cpu", identity=declared)


def identities(loaded):
    return {"adapter": tensors.digest(adapter_state(loaded)), "tokenizer": operation.digest(make_tokenizer()),
            "base": frozen.digest(loaded), "assembly": assembly.digest(loaded, INFERENCE)}


class TeacherScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(TEST_THREADS)
        cls.root = Path(tempfile.mkdtemp(prefix="invar-hf-score-"))
        cls.targets = {}
        for name, declared in (("student", IDENTITY), ("teacher", TEACHER)):
            loaded = MODELS[declared]()
            checkpoint = cls.root / (name + ".safetensors")
            save_file(adapter_state(loaded), checkpoint)
            config = cls.root / (name + ".json")
            config.write_text(json.dumps({"format": FORMAT, "model": declared[0], "revision": declared[1]}))
            cls.targets[name] = (loaded, checkpoint, config, identities(loaded))
        _, checkpoint, _, student = cls.targets["student"]
        cls.source = {"digest": student["adapter"], **{key + "-digest": student[key] for key in ("tokenizer", "base", "assembly")},
                      "prompt": "Compute the answer.", "tokens": 4, "temperature": 0.8, "seed": 41,
                      "call": 7, "attempt": 7, "instance": 7}
        cls.log = cls.root / "source.jsonl"
        completed = cls.run_core(["infer", *flags({**cls.source, "python": sys.executable, "worker": SOURCE_ENTRY,
                                                   "cache": cls.root, "adapter": checkpoint})])
        cls.log.write_text(completed.stdout)
        cls.result = json.loads(completed.stdout.splitlines()[-1])

    @staticmethod
    def run_core(arguments):
        completed = subprocess.run([CORE, *map(str, arguments)], capture_output=True, text=True, timeout=CHILD_SECONDS)
        if completed.returncode:
            raise AssertionError(completed.stderr)
        return completed

    def score(self, name, call):
        _, checkpoint, config, identity = self.targets[name]
        completed = self.run_core(["score", *flags({**{"source-" + key: value for key, value in self.source.items()},
                                                    "source-log": self.log, "source-exit-code": 0,
                                                    "target-digest": identity["adapter"],
                                                    **{"target-" + key + "-digest": identity[key] for key in ("tokenizer", "base", "assembly")},
                                                    "call": call, "attempt": call, "instance": call,
                                                    "python": sys.executable, "worker": SCORE_ENTRY, "cache": self.root,
                                                    "adapter": checkpoint, "worker-config": config})])
        return json.loads(completed.stdout.splitlines()[-1])["observation"]

    def test_bare_teacher_scores_the_admitted_student_path(self):
        teacher, _, _, identity = self.targets["teacher"]
        self.assertEqual(identity["adapter"], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        observed = self.score("teacher", 20)
        self.assertEqual((observed["target"]["model"], observed["target"]["revision"]), TEACHER)
        tokens = torch.tensor([observed["prefix_tokens"] + observed["response_tokens"]])
        self.assertEqual(observed["prefix_tokens"] + observed["response_tokens"], self.result["tokens"])
        direct = scored(teacher, tokens, len(observed["prefix_tokens"]), self.source["temperature"], device="cpu")
        self.assertEqual(observed["log_probability_bits"], [float32.word(value) for value in direct.tolist()])
        self.assertNotEqual(observed["log_probability_bits"], self.result["behavior_bits"])

    def test_student_rescores_its_own_sampled_path_exactly(self):
        observed = self.score("student", 21)
        self.assertEqual(observed["log_probability_bits"], self.result["behavior_bits"])


if __name__ == "__main__":
    unittest.main()
