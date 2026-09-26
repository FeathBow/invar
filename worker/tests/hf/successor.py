import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from peft.tuners.lora.layer import LoraLayer

from worker.hf import assembly
from worker.hf import operation
from worker.hf.learning import update
from worker.tests.hf.learning import batch, make_learner
from worker.hf.model import adapter_state
from worker.hf.tensors import assert_equal, digest
from worker.hf.checkpoint import checkpoint, restore
from worker.hf.step import checkpoint_update
from worker.tests.hf.step import prepared
from worker.implementation import LEARNING

OTHER_IDENTITY = "0" * 64


def changed(learner, tokenizer, field):
    if field == "base":
        with torch.no_grad():
            next(value for value in learner.model.parameters() if not value.requires_grad).add_(1)
    elif field == "assembly":
        layer = next(module for module in learner.model.modules() if isinstance(module, LoraLayer))
        layer.scaling["default"] += 1
    else:
        tokenizer.add_tokens(["previously unused vocabulary entry"])


def observed(learner, tokenizer):
    return (copy.deepcopy(learner.model.state_dict()), copy.deepcopy(learner.optimizer.state_dict()),
            torch.get_rng_state(), torch.cuda.get_rng_state_all(),
            assembly.digest(learner.model, LEARNING), operation.digest(tokenizer))


class SuccessorTests(unittest.TestCase):
    def destination(self):
        return Path(tempfile.mkdtemp(prefix="invar-successor-"))

    def test_bound_checkpoint_preserves_the_actual_update_and_continuation(self):
        learner, _, request, options = prepared()
        result = update(learner, batch())
        output = self.destination()
        state = checkpoint_update(learner, request, output, tokenizer=options.tokenizer, summary=result.summary)
        saved = torch.load(output / "learner.pt", weights_only=True)
        self.assertEqual(saved["adapter"], result.summary["after"])
        self.assertEqual(saved["adapter"], digest(state))
        self.assertEqual({name: saved[name] for name in ("base", "assembly", "tokenizer")},
                         {name: getattr(request, name) for name in ("base", "assembly", "tokenizer")})
        expected = update(learner, batch())
        resumed = make_learner()
        restore(resumed.model, resumed.optimizer, output, tokenizer=options.tokenizer)
        actual = update(resumed, batch())
        assert_equal(expected.summary, actual.summary)
        assert_equal(expected.gradients, actual.gradients)
        assert_equal(adapter_state(learner.model), adapter_state(resumed.model))
        assert_equal(learner.optimizer.state_dict(), resumed.optimizer.state_dict())

    def test_changed_materialization_cannot_write_a_successor_checkpoint(self):
        for field in ("base", "assembly", "tokenizer"):
            with self.subTest(field=field):
                learner, _, request, options = prepared()
                result = update(learner, batch())
                changed(learner, options.tokenizer, field)
                before = observed(learner, options.tokenizer)
                output = self.destination()
                with self.assertRaisesRegex(RuntimeError, "Checkpoint identity differs"):
                    checkpoint_update(learner, request, output, tokenizer=options.tokenizer, summary=result.summary)
                self.assertEqual(list(output.iterdir()), [])
                assert_equal(before, observed(learner, options.tokenizer))

    def test_adapter_changed_after_update_cannot_be_staged(self):
        learner, _, request, options = prepared()
        result = update(learner, batch())
        with torch.no_grad():
            next(value for value in learner.model.parameters() if value.requires_grad).add_(1)
        before = observed(learner, options.tokenizer)
        output = self.destination()
        with self.assertRaisesRegex(RuntimeError, "Checkpoint identity differs"):
            checkpoint_update(learner, request, output, tokenizer=options.tokenizer, summary=result.summary)
        self.assertEqual(list(output.iterdir()), [])
        assert_equal(before, observed(learner, options.tokenizer))

    def test_different_update_identity_cannot_write_checkpoint_files(self):
        learner, _, request, options = prepared()
        result = update(learner, batch())
        before = observed(learner, options.tokenizer)
        for field, message in (("before", "Update input differs"), ("after", "Checkpoint identity differs")):
            with self.subTest(field=field):
                output = self.destination()
                summary = {**result.summary, field: OTHER_IDENTITY}
                with self.assertRaisesRegex(RuntimeError, message):
                    checkpoint_update(learner, request, output, tokenizer=options.tokenizer, summary=summary)
                self.assertEqual(list(output.iterdir()), [])
                assert_equal(before, observed(learner, options.tokenizer))

    def test_checkpoint_expectation_must_be_explicit(self):
        learner, _, _, options = prepared()
        output = self.destination()
        with self.assertRaisesRegex(TypeError, "expected"):
            checkpoint(learner.model, learner.optimizer, output, tokenizer=options.tokenizer)
        self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
