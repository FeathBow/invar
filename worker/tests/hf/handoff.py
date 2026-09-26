import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file

from worker.hf import assembly
from worker.hf import frozen
from worker.hf import operation
from worker.hf.handoff import CONFIG, WEIGHTS, export
from worker.implementation import file_digest
from worker.hf.learning import update
from worker.hf.checkpoint import CheckpointIdentity
from worker.hf.model import adapter_state
from worker.hf.tensors import assert_equal, digest
from worker.tests.hf.learning import batch, make_learner
from worker.tests.hf.decoding import hybrid
from worker.tests.hf.tokenization import make_tokenizer
from worker.implementation import LEARNING


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-handoff-"))
        self.learner = make_learner()
        self.tokenizer = make_tokenizer()
        self.base = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=False))
        with torch.no_grad():
            self.base[0].weight.copy_(self.learner.model.get_base_model()[0].base_layer.weight)

    def identity(self):
        model = self.learner.model
        return CheckpointIdentity(adapter=digest(adapter_state(model)), base=frozen.digest(model),
                                  assembly=assembly.digest(model, LEARNING), tokenizer=operation.digest(self.tokenizer))

    def test_two_real_updates_export_and_reload_exact_fp32_outputs(self):
        prior = None
        for generation in range(2):
            update(self.learner, batch())
            expected = self.identity()
            self.assertNotEqual(prior, expected.adapter)
            prior = expected.adapter
            destination = self.directory / str(generation)
            before_config = copy.deepcopy(self.learner.model.peft_config["default"].to_dict())
            optimizer = copy.deepcopy(self.learner.optimizer.state_dict())
            rng = torch.get_rng_state().clone()
            receipt = export(self.learner.model, destination, tokenizer=self.tokenizer, expected=expected)
            assert_equal(rng, torch.get_rng_state())
            assert_equal(optimizer, self.learner.optimizer.state_dict())
            self.assertEqual(before_config, self.learner.model.peft_config["default"].to_dict())
            config = PeftConfig.from_pretrained(destination, local_files_only=True)
            self.assertEqual(config.r, before_config["r"])
            self.assertEqual(config.lora_alpha, before_config["lora_alpha"])
            self.assertEqual(set(config.target_modules), set(before_config["target_modules"]))
            state = load_file(destination / WEIGHTS)
            self.assertTrue(all(value.dtype == torch.float32 for value in state.values()))
            assert_equal(adapter_state(self.learner.model), state)
            consumer = PeftModel.from_pretrained(copy.deepcopy(self.base), destination,
                                                 local_files_only=True, is_trainable=False)
            assert_equal(state, adapter_state(consumer))
            with torch.no_grad():
                features = torch.tensor([[1.0], [-2.0], [0.5]])
                assert_equal(self.learner.model(features), consumer(features))
            self.assertEqual(receipt, json.loads((destination / "handoff.json").read_text()))
            self.assertEqual(receipt["source"]["adapter"], expected.adapter)
            self.assertEqual(receipt["files"], {name: file_digest(destination / name) for name in (CONFIG, WEIGHTS)})

    def test_wrong_source_identity_fails_before_creating_an_export(self):
        expected = self.identity()
        for name in ("adapter", "base", "assembly", "tokenizer"):
            with self.subTest(identity=name):
                destination = self.directory / name
                with self.assertRaises((ValueError, RuntimeError)):
                    export(self.learner.model, destination, tokenizer=self.tokenizer,
                           expected=replace(expected, **{name: "0" * 64}))
                self.assertFalse(destination.exists())

    def test_existing_destination_is_not_overwritten(self):
        destination = self.directory / "saved"
        export(self.learner.model, destination, tokenizer=self.tokenizer, expected=self.identity())
        before = {path.name: path.read_bytes() for path in destination.iterdir()}
        update(self.learner, batch())
        with self.assertRaises(FileExistsError):
            export(self.learner.model, destination, tokenizer=self.tokenizer, expected=self.identity())
        self.assertEqual(before, {path.name: path.read_bytes() for path in destination.iterdir()})

    def test_merged_adapter_cannot_be_exported_as_unmerged(self):
        self.learner.model.merge_adapter()
        destination = self.directory / "merged"
        with self.assertRaisesRegex(ValueError, "active, unmerged"):
            export(self.learner.model, destination, tokenizer=self.tokenizer, expected=self.identity())
        self.assertFalse(destination.exists())

    def test_configuration_cannot_hide_changed_runtime_scaling(self):
        self.learner.model.get_base_model()[0].scaling["default"] *= 2
        destination = self.directory / "scaling"
        with self.assertRaisesRegex(ValueError, "actual LoRA scaling"):
            export(self.learner.model, destination, tokenizer=self.tokenizer, expected=self.identity())
        self.assertFalse(destination.exists())

    def test_configuration_cannot_hide_changed_target_declarations(self):
        self.learner.model.peft_config["default"].target_modules = {"missing"}
        destination = self.directory / "targets"
        with self.assertRaisesRegex(ValueError, "actual LoRA targets"):
            export(self.learner.model, destination, tokenizer=self.tokenizer, expected=self.identity())
        self.assertFalse(destination.exists())

    def test_qwen_hybrid_exports_every_resolved_mlp_target_without_renaming(self):
        model = hybrid()
        with torch.no_grad():
            for index, parameter in enumerate(value for value in model.parameters() if value.requires_grad):
                parameter.fill_((index + 1) / 100)
        base = copy.deepcopy(model).unload()
        expected = CheckpointIdentity(adapter=digest(adapter_state(model)), base=frozen.digest(model),
                                      assembly=assembly.digest(model, LEARNING), tokenizer=operation.digest(self.tokenizer))
        destination = self.directory / "qwen"
        export(model, destination, tokenizer=self.tokenizer, expected=expected)
        config = PeftConfig.from_pretrained(destination, local_files_only=True)
        self.assertEqual(len(config.target_modules), 6)
        self.assertEqual(config.target_modules, model.peft_config["default"].target_modules)
        state = load_file(destination / WEIGHTS)
        self.assertEqual(len(state), 12)
        consumer = PeftModel.from_pretrained(base, destination, local_files_only=True)
        assert_equal(state, adapter_state(consumer))
        tokens = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            assert_equal(model(input_ids=tokens, use_cache=False).logits,
                         consumer(input_ids=tokens, use_cache=False).logits)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
