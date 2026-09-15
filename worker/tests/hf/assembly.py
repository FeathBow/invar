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
from transformers import Qwen3_5Config

from worker.hf import assembly
from worker.hf import frozen
from worker.tests.hf.checkpoint import advance_square, square_learner
from worker.tests.hf.learning import make_learner
from worker.tests.hf.tokenization import make_tokenizer
from worker.hf.probe import adapter_state, assert_equal, checkpoint, restore
from worker.hf.step import restore_inputs
from worker.tests.hf.step import prepared


def lora(model):
    return next(value for value in model.modules() if isinstance(value, LoraLayer))


class AssemblyTests(unittest.TestCase):
    def test_changed_actual_scaling_is_rejected_before_restoration(self):
        model, optimizer = square_learner()
        destination = Path(tempfile.mkdtemp(prefix="invar-assembly-"))
        checkpoint(model, optimizer, destination, tokenizer=make_tokenizer(), expected=None)
        inputs = torch.ones(1, 2)
        expected = model(inputs).detach().clone()
        base, adapter = frozen.digest(model), adapter_state(model)
        settings = copy.deepcopy(model.peft_config["default"].to_dict())
        lora(model).scaling["default"] *= 2
        self.assertFalse(torch.equal(expected, model(inputs)))
        self.assertEqual(base, frozen.digest(model))
        assert_equal(adapter, adapter_state(model))
        self.assertEqual(settings, model.peft_config["default"].to_dict())
        before, moments = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        with self.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
            restore(model, optimizer, destination, tokenizer=make_tokenizer())
        assert_equal(before, model.state_dict())
        assert_equal(moments, optimizer.state_dict())
        assert_equal(cpu_rng, torch.get_rng_state())
        assert_equal(cuda_rng, torch.cuda.get_rng_state_all())
        lora(model).scaling["default"] /= 2
        advance_square(model, optimizer)
        restore(model, optimizer, destination, tokenizer=make_tokenizer())
        assert_equal(expected, model(inputs).detach())

    def test_declared_and_realized_lora_settings_are_separate_inputs(self):
        model, _ = square_learner()
        original = assembly.digest(model)
        model.peft_config["default"].lora_alpha += 1
        self.assertNotEqual(original, assembly.digest(model))
        model.peft_config["default"].lora_alpha -= 1
        lora(model).lora_dropout["default"] = torch.nn.Dropout(p=0.25)
        dropped = assembly.digest(model)
        self.assertNotEqual(original, dropped)
        lora(model).lora_dropout["default"].p = 0.5
        self.assertNotEqual(dropped, assembly.digest(model))

    def test_attention_choice_omitted_by_transformers_serialization_is_bound(self):
        config = Qwen3_5Config()
        config.text_config._attn_implementation = "eager"
        serialized = config.to_dict()
        original = assembly.configuration(config)
        config.text_config._attn_implementation = "sdpa"
        self.assertEqual(serialized, config.to_dict())
        self.assertNotEqual(original, assembly.configuration(config))

    def test_actual_activation_recomputation_changes_the_assembly_binding(self):
        from worker.tests.hf.decoding import hybrid

        model = hybrid()
        original, base = assembly.digest(model), frozen.digest(model)
        self.assertFalse(model.is_gradient_checkpointing)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertEqual(base, frozen.digest(model))
        with self.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
            assembly.verify(model, original)
        model.gradient_checkpointing_disable()
        self.assertEqual(original, assembly.digest(model))

    def test_configuration_locations_and_transient_train_mode_do_not_define_identity(self):
        model, _ = square_learner()
        config = Qwen3_5Config()
        model.get_base_model().config = config
        original = assembly.digest(model)
        config._name_or_path = "/relocated/model"
        config.text_config._name_or_path = "/relocated/text"
        model.peft_config["default"].base_model_name_or_path = "/relocated/model"
        model.eval()
        self.assertEqual(original, assembly.digest(model))
        config.text_config.rms_norm_eps *= 2
        self.assertNotEqual(original, assembly.digest(model))

    def test_missing_assembly_binding_fails_without_reconstruction(self):
        model, optimizer = square_learner()
        destination = Path(tempfile.mkdtemp(prefix="invar-unbound-assembly-"))
        checkpoint(model, optimizer, destination, tokenizer=make_tokenizer(), expected=None)
        path = destination / "learner.pt"
        saved = torch.load(path, weights_only=True)
        torch.save({key: value for key, value in saved.items() if key != "assembly"}, path)
        with self.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
            restore(model, optimizer, destination, tokenizer=make_tokenizer())

    def test_actual_update_input_path_requires_the_assembly_binding(self):
        _, _, declared, options = prepared()
        consumer = make_learner()
        lora(consumer.model).scaling["default"] *= 2
        before, rng = adapter_state(consumer.model), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
            restore_inputs(consumer.model, declared, options, tokenizer=options.tokenizer)
        assert_equal(before, adapter_state(consumer.model))
        assert_equal(rng, torch.get_rng_state())

    def test_trainable_dtype_mismatch_is_rejected_before_adapter_installation(self):
        model, optimizer = square_learner()
        destination = Path(tempfile.mkdtemp(prefix="invar-assembly-dtype-"))
        checkpoint(model, optimizer, destination, tokenizer=make_tokenizer(), expected=None)
        base = frozen.digest(model)
        advance_square(model, optimizer)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.detach().double()
        self.assertEqual(base, frozen.digest(model))
        before, moments = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
            restore(model, optimizer, destination, tokenizer=make_tokenizer())
        assert_equal(before, model.state_dict())
        assert_equal(moments, optimizer.state_dict())
        assert_equal(rng, torch.get_rng_state())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
