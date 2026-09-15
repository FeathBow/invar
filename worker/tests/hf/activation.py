import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

import torch
from peft.tuners.lora.layer import LoraLayer

from worker.hf import assembly
from worker.hf import frozen
from worker.tests.hf.learning import make_learner
from worker.hf.policy import activate
from worker.hf.probe import adapter_state
from worker.hf.tensors import assert_equal, digest


class ActivationTests(unittest.TestCase):
    def test_declared_materialization_installs_the_actual_adapter(self):
        model = make_learner().model
        state = {name: value + 1 for name, value in adapter_state(model).items()}
        actual = activate(model, state, base=frozen.digest(model), assembly=assembly.digest(model))
        self.assertEqual(actual, digest(state))
        assert_equal(state, adapter_state(model))

    def test_different_frozen_weights_cannot_consume_the_same_adapter(self):
        model = make_learner().model
        expected_base, expected_assembly = frozen.digest(model), assembly.digest(model)
        state = {name: value + 1 for name, value in adapter_state(model).items()}
        with torch.no_grad():
            next(value for value in model.parameters() if not value.requires_grad).add_(1)
        before, rng = adapter_state(model), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "frozen base binding mismatch"):
            activate(model, state, base=expected_base, assembly=expected_assembly)
        assert_equal(before, adapter_state(model))
        assert_equal(rng, torch.get_rng_state())

    def test_different_lora_scaling_cannot_consume_the_same_adapter(self):
        model = make_learner().model
        expected_base, expected_assembly = frozen.digest(model), assembly.digest(model)
        state = {name: value + 1 for name, value in adapter_state(model).items()}
        layer = next(module for module in model.modules() if isinstance(module, LoraLayer))
        layer.scaling["default"] += 1
        before, rng = adapter_state(model), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
            activate(model, state, base=expected_base, assembly=expected_assembly)
        assert_equal(before, adapter_state(model))
        assert_equal(rng, torch.get_rng_state())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
