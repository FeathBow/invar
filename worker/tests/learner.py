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

from cohort import decode as cohort
from .cohort import request
from .inference import IDENTITY, measured, model, TEST_THREADS
from invocation import approve
from learning import parameters
from registry import learning
from probe import adapter_state, assert_equal, checkpoint, digest
from rollout import Request, generate, logprobs, reward
from step import file_digest, optimizer_options, run
from .tokenization import make_tokenizer
from update import decode

REWARDING_SEED = 1326
OTHER_SEED = 41
TOKEN_LIMIT = 4


def fixture():
    directory = Path(tempfile.mkdtemp(prefix="invar-learner-load-"))
    initial = directory / "initial"
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
                        "behavior_bits": actual.behavior.view(torch.uint32).tolist(),
                        "reward": reward(actual.text, "#### 437", actual.truncated)})
    actual = {**request(), "policy": digest(state), "learner": file_digest(initial / "learner.pt"),
              "tokenizer": saved["tokenizer"], "base": saved["base"], "assembly": saved["assembly"],
              "reference": digest(state), "samples": samples, "order": ["s0", "s1"]}
    bound = {"call": 7, "attempt": 11, "instance": 13}
    call = decode({"request": actual, "invocation": {"binding": bound, "program": "update test"},
                   "load": {"binding": bound, "program": "load test"}})
    options = SimpleNamespace(checkpoint=initial, reference=initial / "adapter.safetensors",
                              output=directory / "staged", cache=directory)
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
                evaluate=partial(logprobs, device="cpu"))
        load, = records(output, "loaded_learner")
        consumed, = records(output, "consumed")
        result, = records(output, "result")
        expected = {"binding": call.load.binding(), "program": call.load.program}
        self.assertEqual(load["load"], expected)
        self.assertEqual(load["image"], learning(load["state"]))
        self.assertEqual(consumed["load"], expected)
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
                evaluate=partial(logprobs, device="cpu"))
        self.assertEqual(len(records(output, "consumed")), 1)
        self.assertFalse(records(output, "reward_update"))
        self.assertFalse(records(output, "result"))
        self.assertEqual(list(options.output.iterdir()), [])
        assert_equal(before, adapter_state(loaded))


if __name__ == "__main__":
    torch.set_num_threads(TEST_THREADS)
    unittest.main()
