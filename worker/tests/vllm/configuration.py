from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

from worker.vllm.configuration import CHECKPOINT_FORMAT, FORMAT, configuration, read


def declared():
    return {"format": FORMAT, "model": "invar/local-fixture", "revision": "1" * 40, "handoff": "2" * 64,
            "engine": {"dtype": "bfloat16", "lora_dtype": "float32", "max_model_len": 64}}


class ConfigurationTests(unittest.TestCase):
    def test_launch_owns_an_immutable_configuration_snapshot(self):
        source = declared()
        value = configuration(source)
        source["engine"]["dtype"] = "float16"
        actual = value.arguments()
        self.assertEqual(actual["dtype"], "bfloat16")
        actual["dtype"] = "float16"
        self.assertEqual(value.arguments()["dtype"], "bfloat16")
        with self.assertRaises(FrozenInstanceError):
            value.model = "different/model"

    def test_model_revision_handoff_and_location_overrides_are_checked(self):
        changes = ({"model": ""}, {"revision": "main"}, {"handoff": "unbound"},
                   {"engine": {"model": "different/model"}}, {"engine": {"temperature": float("nan")}})
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                configuration({**declared(), **changed})

    def test_duplicate_configuration_fields_fail_at_the_file_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            path.write_text(json.dumps(declared()))
            self.assertEqual(read(path), configuration(declared()))
            path.write_text(path.read_text().replace('"dtype": "bfloat16"', '"dtype": "bfloat16", "dtype": "float16"'))
            with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                read(path)

    def test_checkpoint_launch_binds_a_template_and_resolves_its_relative_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            value = {**declared(), "format": CHECKPOINT_FORMAT, "template": "configuration-source"}
            path.write_text(json.dumps(value))
            actual = read(path)
            self.assertEqual(actual.template, str(path.parent.resolve() / "configuration-source"))
            self.assertEqual(actual.handoff, value["handoff"])
            self.assertEqual(actual.arguments(), value["engine"])
            self.assertIsNone(configuration(declared()).template)

    def test_checkpoint_mode_requires_its_explicit_template_schema(self):
        invalid = ({**declared(), "format": CHECKPOINT_FORMAT},
                   {**declared(), "format": CHECKPOINT_FORMAT, "template": ""},
                   {**declared(), "format": CHECKPOINT_FORMAT, "template": True},
                   {**declared(), "template": "unexpected"})
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                configuration(value)
