import unittest

try:
    import mlx  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

import mlx.core as mx

from worker.mlx import metrics as mlx_metrics


class MetricsTests(unittest.TestCase):
    def test_stage_without_new_allocation_retains_its_active_memory_peak(self):
        value = mx.ones((1024,), dtype=mx.float32)
        mx.eval(value)
        required = value.nbytes
        observed = []
        returned = mlx_metrics.measure("checkpoint", lambda: value, emit=lambda stage, values: observed.append((stage, values)))
        self.assertIs(returned, value)
        self.assertEqual(observed[0][0], "checkpoint")
        self.assertEqual(observed[0][1]["allocator"], "mlx")
        self.assertGreaterEqual(observed[0][1]["peak_active"], required)


if __name__ == "__main__":
    unittest.main()
