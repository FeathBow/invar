import unittest

import torch

from frozen import HASH_CHUNK_BYTES, blocks, digest


class FrozenTests(unittest.TestCase):
    def test_trainable_aliases_are_excluded_but_frozen_parameters_and_buffers_are_bound(self):
        model = torch.nn.Module()
        model.register_parameter("adapter", torch.nn.Parameter(torch.ones(1)))
        model.register_parameter("alias", model.adapter)
        model.register_parameter("base", torch.nn.Parameter(torch.ones(1), requires_grad=False))
        model.register_buffer("quantization", torch.zeros(1, dtype=torch.uint8))
        original = digest(model)
        with torch.no_grad():
            model.adapter.add_(1)
        self.assertEqual(original, digest(model))
        with torch.no_grad():
            model.base.neg_()
        changed = digest(model)
        self.assertNotEqual(original, changed)
        model.quantization.add_(1)
        self.assertNotEqual(changed, digest(model))

    def test_tensor_names_dtypes_shapes_and_signed_zero_are_bound(self):
        originals = []
        for name, value in (("base", torch.tensor([0.])), ("other", torch.tensor([0.])),
                            ("base", torch.tensor([0.], dtype=torch.float64)),
                            ("base", torch.tensor([[0.]])), ("base", torch.tensor([-0.]))):
            model = torch.nn.Module()
            model.register_buffer(name, value)
            originals.append(digest(model))
        self.assertEqual(len(set(originals)), len(originals))

    def test_large_noncontiguous_tensor_stream_matches_contiguous_bytes(self):
        width = HASH_CHUNK_BYTES // torch.tensor(0, dtype=torch.int32).element_size()
        value = torch.arange(width * 2, dtype=torch.int32).reshape(width, 2).t()
        self.assertFalse(value.is_contiguous())
        pieces = list(blocks(value))
        self.assertGreater(len(pieces), 1)
        self.assertLessEqual(max(map(len, pieces)), HASH_CHUNK_BYTES)
        self.assertEqual(b"".join(pieces), value.contiguous().view(torch.uint8).numpy().tobytes())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
