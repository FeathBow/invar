import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from worker.cohort import decode as cohort
from worker.tests.hf.cohort import request
from worker.tests.hf.inference import IDENTITY, measured, model, TEST_THREADS
from worker.invocation import approve
from worker.hf.learning import parameters
from worker.registry import learning
from worker.hf.model import adapter_state
from worker.hf.tensors import assert_equal, digest
from worker.hf.checkpoint import checkpoint
from worker.hf.probability import logprobs
from worker.hf.rollout import generate
from worker.trajectory import Request
from worker.hf.step import file_digest, optimizer_options, run
from worker.tests.hf.tokenization import make_tokenizer
from worker.update import decode
from worker.tests.hf.responder import receiver, request_core

REWARDING_SEED = 1326
OTHER_SEED = 41
TOKEN_LIMIT = 4


def fixture(*, directory=None):
    root = Path(tempfile.mkdtemp(prefix="invar-learner-load-")) if directory is None else directory
    if directory is not None:
        root.mkdir(exist_ok=False)
    initial = root / "initial"
    initial.mkdir()
    loaded, tokenizer = model(), make_tokenizer()
    optimizer = torch.optim.AdamW(parameters(loaded), **optimizer_options(cohort(request()).optimizer))
    state = checkpoint(loaded, optimizer, initial, tokenizer=tokenizer, expected=None)
    saved = torch.load(initial / "learner.pt", weights_only=True)
    samples = []
    for index, seed in enumerate((REWARDING_SEED, OTHER_SEED)):
        numerical = Request(sample=f"s{index}", group="g0", prompt="Compute the answer.",
                            seed=seed, limit=TOKEN_LIMIT, temperature=0.8)
        actual = generate(loaded, tokenizer, numerical, device="cpu")
        samples.append({**asdict(numerical), "tokens": actual.tokens[0].tolist(),
                        "prompt_length": actual.prompt_length, "text": actual.text,
                        "truncated": actual.truncated,
                        "behavior_bits": actual.behavior.view(torch.uint32).tolist(), "reference_bits": [],
                        "reward": (1.0, 0.0)[index],
                        "advantage_bits": (0x3F7FF2E5, 0xBF7FF2E5)[index]})
    actual = {**request(), "policy": digest(state), "learner": file_digest(initial / "learner.pt"),
              "tokenizer": saved["tokenizer"], "base": saved["base"], "assembly": saved["assembly"],
              "behavior_model": {name: saved[name] for name in ("base", "assembly")},
              "reference": digest(state), "samples": samples, "order": ["s0", "s1"], "steps": [["s0", "s1"]]}
    bound = {"call": 7, "attempt": 11, "instance": 13}
    call = decode({"request": actual, "invocation": {"binding": bound, "program": "update test"},
                   "load": {"binding": bound, "program": "load test"}})
    options = SimpleNamespace(checkpoint=initial, reference=initial / "adapter.safetensors",
                              output=root / "staged", cache=root)
    return call, options


def records(output, stage):
    return [value for line in output.getvalue().splitlines()
            if (value := json.loads(line))["stage"] == stage]


class LearnerLoadTests(unittest.TestCase):
    def test_actual_restoration_and_consumption_precede_reward_update(self):
        call, options = fixture()
        self.assertEqual([item.reward for item in call.request.samples], [1, 0])
        loaded = model()
        permission = io.StringIO(json.dumps({"binding": call.invocation.binding(),
                                            "program": call.invocation.program}) + "\n")
        output = io.StringIO()
        with redirect_stdout(output):
            run(call, options, loader=lambda options, request: (loaded, make_tokenizer(), IDENTITY),
                measure=measured, permission=partial(approve, source=permission),
                evaluate=partial(logprobs, device="cpu"), receive=receiver(output, request_core(call.request)))
        load, = records(output, "loaded_learner")
        consumed, = records(output, "consumed")
        result, = records(output, "result")
        expected = {"binding": call.load.binding(), "program": call.load.program}
        self.assertEqual(load["load"], expected)
        self.assertEqual(load["image"], learning(load["state"]))
        self.assertEqual(consumed["load"], expected)
        self.assertEqual(consumed["request"]["behavior_model"], asdict(call.request.behavior_model))
        self.assertEqual(consumed["request"]["base"], call.request.base)
        self.assertEqual(consumed["request"]["assembly"], call.request.assembly)
        self.assertEqual([item["behavior_bits"] for item in consumed["request"]["samples"]],
                         [list(item.behavior_bits) for item in call.request.samples])
        self.assertEqual(result["update"]["before"], call.request.policy)
        self.assertNotEqual(result["adapter"], call.request.policy)
        self.assertGreater(result["update"]["reward_gradient_norm"], 0)
        saved = torch.load(options.output / "learner.pt", weights_only=True)
        self.assertEqual(saved["adapter"], result["adapter"])
        self.assertTrue(all(slot["step"].item() == 1 for slot in saved["optimizer"]["state"].values()))

    def test_rejected_permission_preserves_parameters_and_writes_no_successor(self):
        call, options = fixture()
        loaded = model()
        before = adapter_state(loaded)
        permission = io.StringIO(json.dumps({"binding": call.invocation.binding(), "program": "wrong"}) + "\n")
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "permission differs"):
            run(call, options, loader=lambda options, request: (loaded, make_tokenizer(), IDENTITY),
                measure=measured, permission=partial(approve, source=permission),
                evaluate=partial(logprobs, device="cpu"), receive=receiver(output, request_core(call.request)))
        self.assertEqual(len(records(output, "consumed")), 1)
        self.assertFalse(records(output, "reward_update"))
        self.assertFalse(records(output, "result"))
        self.assertEqual(list(options.output.iterdir()), [])
        assert_equal(before, adapter_state(loaded))


if __name__ == "__main__":
    torch.set_num_threads(TEST_THREADS)
    unittest.main()
