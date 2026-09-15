import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

import torch

from worker.vllm.identity import base, digest


class NativeIdentityTests(unittest.TestCase):
    def test_actual_weight_and_nonpersistent_buffer_bytes_are_bound(self):
        model = torch.nn.Linear(3, 2)
        model.register_buffer("positions", torch.arange(4), persistent=False)
        before = base(model)
        saved = model.weight.detach().clone()
        with torch.no_grad():
            model.weight[0, 0] += 1
        self.assertNotEqual(before, base(model))
        with torch.no_grad():
            model.weight.copy_(saved)
        self.assertEqual(before, base(model))
        model.positions[0] += 1
        self.assertNotEqual(before, base(model))

    def test_identity_rejects_ambiguous_or_nonfinite_metadata(self):
        with self.assertRaisesRegex(ValueError, "keys alias"):
            digest({1: "first", "1": "second"})
        with self.assertRaises(ValueError):
            digest({"temperature": float("nan")})

    def test_tensor_views_preserve_values_and_bind_dtype_and_shape(self):
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
        self.assertEqual(digest(source), digest(source.contiguous()))
        self.assertNotEqual(digest(source), digest(source.double()))
        self.assertNotEqual(digest(source), digest(source.reshape(-1)))
