import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from policy import read_adapter, validate_schema
from tensors import digest


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-policy-"))
        self.state = {"adapter": torch.tensor([1.0, -0.0], dtype=torch.float32)}

    def write(self, state, name):
        path = self.directory / name
        save_file(state, path)
        return path

    def test_requested_tensor_content_is_required(self):
        path = self.write(self.state, "adapter.safetensors")
        self.assertEqual(digest(read_adapter(path, digest(self.state))), digest(self.state))
        other = {"adapter": torch.tensor([1.0, 0.0], dtype=torch.float32)}
        with self.assertRaisesRegex(ValueError, "requested tensor identity"):
            read_adapter(path, digest(other))
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            read_adapter(path, "not-a-digest")

    def test_invalid_adapter_contents_are_rejected_before_loading(self):
        cases = [{}, {"adapter": torch.tensor([float("nan")])},
                 {"adapter": torch.tensor([float("inf")])},
                 {"adapter": torch.tensor([1.0], dtype=torch.float64)}]
        for index, state in enumerate(cases):
            with self.subTest(index=index):
                path = self.write(state, f"invalid-{index}.safetensors")
                with self.assertRaisesRegex(ValueError, "finite FP32"):
                    read_adapter(path, digest(state))

    def test_schema_binds_exact_targets_shapes_and_types(self):
        validate_schema(self.state, self.state)
        cases = [{}, {"other": self.state["adapter"]},
                 {"adapter": torch.ones(1)},
                 {"adapter": self.state["adapter"].reshape(1, 2)},
                 {"adapter": self.state["adapter"].double()}]
        for state in cases:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    validate_schema(self.state, state)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
