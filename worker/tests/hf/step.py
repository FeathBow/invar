import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from worker.cohort import decode
from worker.hf.learning import update
from worker.tests.hf.learning import batch, make_learner
from worker.hf.model import adapter_state
from worker.hf.tensors import assert_equal, digest
from worker.hf.checkpoint import checkpoint
from worker.hf.step import file_digest, restore_inputs
from worker.tests.hf.cohort import synchronous
from worker.tests.hf.tokenization import admitted_request, make_tokenizer


def prepared():
    directory = Path(tempfile.mkdtemp(prefix="invar-inputs-"))
    original = make_learner()
    reference = adapter_state(original.model)
    reference_path = directory / "reference.safetensors"
    save_file(reference, reference_path)
    update(original, batch())
    tokenizer = make_tokenizer()
    policy = checkpoint(original.model, original.optimizer, directory, tokenizer=tokenizer, expected=None)
    saved = torch.load(directory / "learner.pt", weights_only=True)
    declared = decode(synchronous({**admitted_request(tokenizer), "policy": digest(policy),
                       "learner": file_digest(directory / "learner.pt"),
                       "base": saved["base"], "assembly": saved["assembly"],
                       "reference": digest(reference)}))
    options = SimpleNamespace(checkpoint=directory, reference=reference_path, tokenizer=tokenizer)
    return original, reference, declared, options


class InputTests(unittest.TestCase):
    def test_bound_inputs_restore_distinct_policy_reference_and_optimizer(self):
        original, reference, declared, options = prepared()
        consumer = make_learner()
        optimizer, fixed, loaded = restore_inputs(consumer.model, declared, options, tokenizer=options.tokenizer)
        self.assertNotEqual(declared.policy, declared.reference)
        assert_equal(adapter_state(original.model), adapter_state(consumer.model))
        assert_equal(original.optimizer.state_dict(), optimizer.state_dict())
        assert_equal(reference, fixed)
        self.assertEqual(loaded, {"policy": declared.policy, "learner": declared.learner,
                                 "reference": declared.reference, "tokenizer": declared.tokenizer,
                                 "base": declared.base, "assembly": declared.assembly,
                                 "optimizer": {"learning_rate": declared.optimizer.learning_rate,
                                               "betas": declared.optimizer.betas,
                                               "epsilon": declared.optimizer.epsilon,
                                               "weight_decay": declared.optimizer.weight_decay}})

    def test_wrong_learner_identity_fails_before_model_or_rng_mutation(self):
        _, _, declared, options = prepared()
        consumer = make_learner()
        before = adapter_state(consumer.model)
        rng = torch.get_rng_state()
        wrong = replace(declared, learner="0" * 64)
        with self.assertRaisesRegex(ValueError, "requested checkpoint bytes"):
            restore_inputs(consumer.model, wrong, options, tokenizer=options.tokenizer)
        assert_equal(before, adapter_state(consumer.model))
        assert_equal(rng, torch.get_rng_state())

    def test_declared_materialization_must_match_the_saved_learner(self):
        _, _, declared, options = prepared()
        consumer = make_learner()
        before, rng = adapter_state(consumer.model), torch.get_rng_state()
        for name in ("base", "assembly"):
            with self.subTest(field=name):
                wrong = replace(declared, **{name: "0" * 64})
                with self.assertRaisesRegex(ValueError, "materialization differs from the requested update input"):
                    restore_inputs(consumer.model, wrong, options, tokenizer=options.tokenizer)
                assert_equal(before, adapter_state(consumer.model))
                assert_equal(rng, torch.get_rng_state())


    def test_bound_update_rejects_different_actual_frozen_weights(self):
        _, _, declared, options = prepared()
        consumer = make_learner()
        with torch.no_grad():
            next(value for value in consumer.model.parameters() if not value.requires_grad).add_(1)
        before, rng = adapter_state(consumer.model), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "frozen base binding mismatch"):
            restore_inputs(consumer.model, declared, options, tokenizer=options.tokenizer)
        assert_equal(before, adapter_state(consumer.model))
        assert_equal(rng, torch.get_rng_state())


    def test_text_mismatch_is_rejected_before_learner_restoration(self):
        _, _, declared, options = prepared()
        consumer = make_learner()
        before, rng = adapter_state(consumer.model), torch.get_rng_state()
        wrong = replace(declared, samples=(replace(declared.samples[0], text="Different answer"),
                                            *declared.samples[1:]))
        with self.assertRaisesRegex(ValueError, "Observed text differs"):
            restore_inputs(consumer.model, wrong, options, tokenizer=options.tokenizer)
        assert_equal(before, adapter_state(consumer.model))
        assert_equal(rng, torch.get_rng_state())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
