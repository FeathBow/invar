import unittest

try:
    import mlx  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

import mlx.core as mx
from mlx_lm.models import gated_delta

from worker.mlx import recurrence as mlx_recurrence
from worker.mlx import tensors as mlx_tensors


class RecurrenceTests(unittest.TestCase):
    def test_segments_preserve_final_state_and_full_reverse_dependency(self):
        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=str(dtype)):
                self.compare(dtype)

    def compare(self, dtype):
        mx.random.seed(29)
        count = mlx_recurrence.CHECKPOINT_TOKENS * 2 + 1
        q = (mx.random.normal((1, count, 2, 4)) / 8).astype(dtype)
        k = (mx.random.normal(q.shape) / 8).astype(dtype)
        v = mx.random.normal((1, count, 4, 4)).astype(dtype)
        g = mx.full((1, count, 4), 0.75)
        beta = mx.full(g.shape, 0.5, dtype=dtype)
        initial = mx.random.normal((1, 4, 4, 4)) / 8
        mask = mx.array([[index % 5 != 0 for index in range(count)]])
        inputs = [q, k, v, g, beta, initial]
        cotangents = [mx.ones_like(v), mx.ones_like(initial)]
        expected, original = mx.vjp(lambda *values: gated_delta.gated_delta_ops(*values, mask=mask), inputs, cotangents)
        observed, segmented = mx.vjp(lambda *values: mlx_recurrence.segmented(*values, mask=mask), inputs, cotangents)
        mx.eval(expected, original, observed, segmented)
        self.assertTrue(mlx_tensors.equal(dict(enumerate(expected)), dict(enumerate(observed))))
        self.assertTrue(mlx_tensors.equal(dict(enumerate(original)), dict(enumerate(segmented))))
        self.assertTrue(mx.any(segmented[-1] != 0).item())
        self.assertTrue(mx.any(segmented[0][:, 1] != 0).item())


if __name__ == "__main__":
    unittest.main()
