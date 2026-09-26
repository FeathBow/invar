import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from worker.hf import assembly
from worker.hf import frozen
from worker.hf import operation
from worker.hf.handoff import export
from worker.hf.learning import update
from worker.implementation import file_digest
from worker.hf.artifact import CONFIG, RECEIPT, WEIGHTS, read, read_checkpoint
from worker.hf.checkpoint import CheckpointIdentity
from worker.hf.model import adapter_state
from worker.hf.tensors import assert_equal, digest
from worker.tests.hf.learning import batch, make_learner
from worker.tests.hf.tokenization import make_tokenizer
from worker.implementation import LEARNING


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-peft-reader-")) / "package"
        self.learner = make_learner()
        self.model = self.learner.model
        self.tokenizer = make_tokenizer()
        expected = CheckpointIdentity(adapter=digest(adapter_state(self.model)), base=frozen.digest(self.model),
                                      assembly=assembly.digest(self.model, LEARNING), tokenizer=operation.digest(self.tokenizer))
        self.receipt = export(self.model, self.directory, tokenizer=self.tokenizer, expected=expected)
        self.identity = file_digest(self.directory / RECEIPT)

    def reseal(self):
        (self.directory / RECEIPT).write_text(json.dumps(self.receipt, sort_keys=True))
        return file_digest(self.directory / RECEIPT)

    def test_reads_actual_export_and_holds_a_snapshot_without_mutating_the_source(self):
        package = read(self.directory, expected=self.identity)
        self.assertEqual(package.identity, self.identity)
        self.assertEqual(dict(package.source), self.receipt["source"])
        assert_equal(adapter_state(self.model), dict(package.tensors))
        package.verify()
        configured = package.configuration
        (self.directory / CONFIG).write_text("{}")
        self.assertEqual(package.configuration, configured)
        with self.assertRaises(TypeError):
            package.source["adapter"] = "0" * 64
        with self.assertRaises(TypeError):
            package.tensors["additional"] = torch.zeros(1)

    def test_changed_receipt_is_rejected_against_the_requested_identity(self):
        self.receipt["source"]["adapter"] = "0" * 64
        self.reseal()
        with self.assertRaisesRegex(ValueError, "file binding mismatch"):
            read(self.directory, expected=self.identity)

    def test_checks_configuration_and_tensor_file_bytes(self):
        for name in (CONFIG, WEIGHTS):
            with self.subTest(file=name):
                path = self.directory / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "file binding mismatch"):
                    read(self.directory, expected=self.identity)
                path.write_bytes(original)

    def test_a_valid_file_hash_cannot_hide_changed_policy_tensors(self):
        values = adapter_state(self.model)
        name = next(iter(values))
        values[name] = values[name] + 1
        save_file(values, self.directory / WEIGHTS)
        self.receipt["files"][WEIGHTS] = file_digest(self.directory / WEIGHTS)
        with self.assertRaisesRegex(ValueError, "requested tensor identity"):
            read(self.directory, expected=self.reseal())

    def test_partial_source_binding_and_half_precision_are_rejected(self):
        self.receipt["source"].pop("base")
        with self.assertRaisesRegex(ValueError, "all four source"):
            read(self.directory, expected=self.reseal())
        self.receipt["source"]["base"] = frozen.digest(self.model)
        values = {name: value.half() for name, value in adapter_state(self.model).items()}
        save_file(values, self.directory / WEIGHTS)
        self.receipt["files"][WEIGHTS] = file_digest(self.directory / WEIGHTS)
        self.receipt["source"]["adapter"] = digest(values)
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            read(self.directory, expected=self.reseal())

    def test_verification_rechecks_actual_tensor_words(self):
        package = read(self.directory, expected=self.identity)
        name = next(iter(package.tensors))
        package.tensors[name].add_(1)
        with self.assertRaisesRegex(ValueError, "requested tensor identity"):
            package.verify()

    def test_checkpoint_configuration_reuse_reads_two_real_updated_policies(self):
        original = read(self.directory, expected=self.identity)
        template = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        prior = original.source["adapter"]
        for generation in range(2):
            updated = update(self.learner, batch())
            self.assertGreater(updated.summary["reward_gradient_norm"], 0)
            state = adapter_state(self.model)
            path = self.directory.parent / f"adapter-{generation}.safetensors"
            save_file(state, path)
            package = read_checkpoint(path, template=self.directory, expected=self.identity)
            self.assertNotEqual(package.identity, self.identity)
            self.assertNotEqual(package.source["adapter"], prior)
            self.assertEqual(package.source["adapter"], updated.summary["after"])
            self.assertEqual(dict(package.source), {**original.source, "adapter": digest(state)})
            self.assertEqual(package.configuration, original.configuration)
            assert_equal(state, dict(package.tensors))
            self.assertEqual(template, {path.name: path.read_bytes() for path in self.directory.iterdir()})
            prior = package.source["adapter"]
            path.write_bytes(b"changed after the snapshot")
            package.verify()
            assert_equal(state, dict(package.tensors))

    def test_checkpoint_requires_the_complete_template_schema_and_fp32_words(self):
        state = adapter_state(self.model)
        name = next(iter(state))
        changes = ({key: value for key, value in state.items() if key != name},
                   {**state, name: state[name].reshape(-1)},
                   {**state, name: state[name].half()},
                   {**state, name: torch.full_like(state[name], float("nan"))})
        path = self.directory.parent / "invalid.safetensors"
        for values in changes:
            save_file(values, path)
            with self.subTest(keys=values.keys()), self.assertRaises(ValueError):
                read_checkpoint(path, template=self.directory, expected=self.identity)

    def test_checkpoint_binding_uses_canonical_policy_words_and_the_actual_template(self):
        state = adapter_state(self.model)
        first = self.directory.parent / "first.safetensors"
        second = self.directory.parent / "second.safetensors"
        save_file(state, first, metadata={"container": "one"})
        save_file(state, second, metadata={"container": "two"})
        self.assertNotEqual(first.read_bytes(), second.read_bytes())
        before = read_checkpoint(first, template=self.directory, expected=self.identity)
        after = read_checkpoint(second, template=self.directory, expected=self.identity)
        self.assertEqual(before.identity, after.identity)
        (self.directory / CONFIG).write_text("{}")
        with self.assertRaisesRegex(ValueError, "file binding mismatch"):
            read_checkpoint(first, template=self.directory, expected=self.identity)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
