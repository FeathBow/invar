import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from worker.hf.learning import update
from worker.tests.hf.learning import batch, make_learner
from worker.tests.hf.tokenization import make_tokenizer
from worker.hf.model import DEFAULT_SEED, adapter_state
from worker.hf.tensors import assert_equal
from worker.hf.checkpoint import checkpoint, restore

SQUARE_WIDTH = 2


def square_learner():
    torch.manual_seed(DEFAULT_SEED)
    base = torch.nn.Sequential(torch.nn.Linear(SQUARE_WIDTH, SQUARE_WIDTH, bias=False))
    model = get_peft_model(base, LoraConfig(r=SQUARE_WIDTH, lora_alpha=SQUARE_WIDTH, target_modules=["0"]))
    parameters = tuple(value for value in model.parameters() if value.requires_grad)
    optimizer = torch.optim.AdamW(parameters, foreach=False, fused=False)
    advance_square(model, optimizer)
    return model, optimizer


def advance_square(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, SQUARE_WIDTH)).square().sum().backward()
    optimizer.step()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-checkpoint-"))

    def destination(self, name):
        path = self.directory / name
        path.mkdir()
        return path

    def test_mixed_policy_and_optimizer_are_rejected_before_mutation(self):
        source = make_learner()
        first, second, mixed = (self.destination(name) for name in ("first", "second", "mixed"))
        update(source, batch())
        checkpoint(source.model, source.optimizer, first, tokenizer=make_tokenizer(), expected=None)
        update(source, batch())
        checkpoint(source.model, source.optimizer, second, tokenizer=make_tokenizer(), expected=None)
        shutil.copyfile(first / "adapter.safetensors", mixed / "adapter.safetensors")
        shutil.copyfile(second / "learner.pt", mixed / "learner.pt")
        consumer = make_learner()
        before = adapter_state(consumer.model)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        with self.assertRaisesRegex(RuntimeError, "Checkpoint adapter binding mismatch"):
            restore(consumer.model, consumer.optimizer, mixed, tokenizer=make_tokenizer())
        assert_equal(before, adapter_state(consumer.model))
        self.assertEqual(consumer.optimizer.state_dict()["state"], {})
        assert_equal(cpu_rng, torch.get_rng_state())
        assert_equal(cuda_rng, torch.cuda.get_rng_state_all())

    def test_fresh_learner_restores_state_and_continues_identically(self):
        original = make_learner()
        update(original, batch())
        destination = self.destination("saved")
        saved = checkpoint(original.model, original.optimizer, destination, tokenizer=make_tokenizer(), expected=None)
        saved_optimizer = torch.load(destination / "learner.pt", weights_only=True)
        expected_result = update(original, batch())
        expected_adapter = adapter_state(original.model)
        consumer = make_learner()
        restore(consumer.model, consumer.optimizer, destination, tokenizer=make_tokenizer())
        assert_equal(saved, adapter_state(consumer.model))
        assert_equal(saved_optimizer["optimizer"], consumer.optimizer.state_dict())
        assert_equal(saved_optimizer["cpu_rng"], torch.get_rng_state())
        assert_equal(saved_optimizer["cuda_rng"], torch.cuda.get_rng_state_all())
        actual_result = update(consumer, batch())
        assert_equal(expected_adapter, adapter_state(consumer.model))
        assert_equal(original.optimizer.state_dict(), consumer.optimizer.state_dict())
        assert_equal(expected_result.summary, actual_result.summary)
        assert_equal(expected_result.gradients, actual_result.gradients)

    def test_same_shape_parameter_reorder_preserves_actual_continuation(self):
        model, producer = square_learner()
        destination = self.destination("named")
        checkpoint(model, producer, destination, tokenizer=make_tokenizer(), expected=None)
        parameters = tuple(value for value in model.parameters() if value.requires_grad)
        self.assertEqual(parameters[0].shape, parameters[1].shape)
        advance_square(model, producer)
        expected = adapter_state(model)
        consumer = torch.optim.AdamW(parameters[::-1], foreach=False, fused=False)
        restore(model, consumer, destination, tokenizer=make_tokenizer())
        advance_square(model, consumer)
        assert_equal(expected, adapter_state(model))
        for parameter in parameters:
            assert_equal(producer.state[parameter], consumer.state[parameter])

    def test_missing_or_ambiguous_binding_fails_before_restoration(self):
        model, producer = square_learner()
        destination = self.destination("invalid")
        checkpoint(model, producer, destination, tokenizer=make_tokenizer(), expected=None)
        path = destination / "learner.pt"
        saved = torch.load(path, weights_only=True)
        name = saved["parameters"][0]
        changes = [{key: value for key, value in saved.items() if key != "parameters"},
                   {**saved, "parameters": {0: name, 1: name}},
                   {**saved, "parameters": {False: name, 1: saved["parameters"][1]}},
                   {**saved, "optimizer": {**saved["optimizer"],
                    "param_groups": [{**saved["optimizer"]["param_groups"][0], "params": [0, 0]}]}}]
        before, rng = adapter_state(model), torch.get_rng_state()
        for candidate in changes:
            with self.subTest(parameters=candidate.get("parameters")):
                torch.save(candidate, path)
                consumer = torch.optim.AdamW(producer.param_groups[0]["params"], foreach=False, fused=False)
                with self.assertRaisesRegex(RuntimeError, "parameter binding"):
                    restore(model, consumer, destination, tokenizer=make_tokenizer())
                assert_equal(before, adapter_state(model))
                assert_equal(rng, torch.get_rng_state())
                self.assertEqual(consumer.state_dict()["state"], {})

    def test_snapshot_requires_the_complete_trainable_parameter_inventory(self):
        learner = make_learner()
        destination = self.destination("incomplete")
        learner.optimizer.param_groups[0]["params"] = learner.optimizer.param_groups[0]["params"][:-1]
        with self.assertRaisesRegex(RuntimeError, "parameter binding"):
            checkpoint(learner.model, learner.optimizer, destination, tokenizer=make_tokenizer(), expected=None)
        self.assertEqual(list(destination.iterdir()), [])

    def test_changed_frozen_base_fails_before_any_restore_mutation(self):
        model, producer = square_learner()
        destination = self.destination("base")
        checkpoint(model, producer, destination, tokenizer=make_tokenizer(), expected=None)
        advance_square(model, producer)
        with torch.no_grad():
            next(value for value in model.parameters() if not value.requires_grad).add_(1)
        before = copy.deepcopy(model.state_dict())
        optimizer = copy.deepcopy(producer.state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        with self.assertRaisesRegex(RuntimeError, "frozen base binding mismatch"):
            restore(model, producer, destination, tokenizer=make_tokenizer())
        assert_equal(before, model.state_dict())
        assert_equal(optimizer, producer.state_dict())
        assert_equal(cpu_rng, torch.get_rng_state())
        assert_equal(cuda_rng, torch.cuda.get_rng_state_all())

    def test_missing_base_binding_is_rejected_without_inference(self):
        model, producer = square_learner()
        destination = self.destination("unbound-base")
        checkpoint(model, producer, destination, tokenizer=make_tokenizer(), expected=None)
        path = destination / "learner.pt"
        saved = torch.load(path, weights_only=True)
        torch.save({key: value for key, value in saved.items() if key != "base"}, path)
        before = copy.deepcopy(producer.state_dict())
        with self.assertRaisesRegex(RuntimeError, "frozen base binding mismatch"):
            restore(model, producer, destination, tokenizer=make_tokenizer())
        assert_equal(before, producer.state_dict())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
