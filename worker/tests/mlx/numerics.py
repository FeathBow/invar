import unittest

try:
    import mlx  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear

from worker.mlx import adapter as mlx_adapter
from worker.mlx import numerics as mlx_numerics
from worker.mlx.projection import SPLIT_ROWS, ColumnLoRALinear, RowLinear
from worker.mlx import rollout as mlx_rollout
from worker.mlx import tensors as mlx_tensors
from worker.tests.mlx.rollout import model
from worker.mlx import training as mlx_training

INPUT_WIDTH = 5120
OUTPUT_WIDTH = 48
ROW_COUNT = SPLIT_ROWS - 1
ROW_COUNTS = (1, 2, 3, 4, 5, 7, 8, 9, 13, 31, 32, 33, ROW_COUNT)
STOCK_ROWS = (SPLIT_ROWS, SPLIT_ROWS + 3, 134)
PARAMETER_SCALE = 0.01


def projection():
    linear = nn.Linear(INPUT_WIDTH, OUTPUT_WIDTH, bias=False)
    linear.set_dtype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)


class NumericsTests(unittest.TestCase):
    def rows(self, operation, inputs):
        expected = operation(inputs)
        mx.eval(expected)
        for count in ROW_COUNTS:
            with self.subTest(rows=count):
                self.assertTrue(mlx_tensors.equal({"output": operation(inputs[:count])}, {"output": expected[:count]}))
        self.assertTrue(mlx_tensors.equal({"output": operation(inputs[::-1])[::-1]}, {"output": expected}))
        middle = ROW_COUNT // 2
        split = mx.concatenate((operation(inputs[:middle]), operation(inputs[middle:])))
        self.assertTrue(mlx_tensors.equal({"output": split}, {"output": expected}))

    def test_nonzero_quantized_and_lora_projections_preserve_physical_rows(self):
        mx.random.seed(71)
        inputs = mx.random.normal((ROW_COUNT, INPUT_WIDTH)).astype(mx.bfloat16)
        linear = projection()
        object.__setattr__(linear, "__class__", RowLinear)
        self.rows(linear, inputs)
        adapter = LoRALinear.from_base(linear, r=mlx_adapter.RANK, scale=mlx_adapter.SCALE)
        adapter.lora_b = mx.random.normal(adapter.lora_b.shape) * PARAMETER_SCALE
        object.__setattr__(adapter, "__class__", ColumnLoRALinear)
        self.rows(adapter, inputs)

    def test_calls_of_stock_rows_use_the_library_projection(self):
        mx.random.seed(79)
        inputs = mx.random.normal((max(STOCK_ROWS), INPUT_WIDTH)).astype(mx.bfloat16)
        linear = projection()
        object.__setattr__(linear, "__class__", RowLinear)
        for count in STOCK_ROWS:
            with self.subTest(rows=count):
                self.assertTrue(mlx_tensors.equal({"output": linear(inputs[:count])},
                                                  {"output": nn.QuantizedLinear.__call__(linear, inputs[:count])}))

    def test_row_independent_generation_prefills_one_request_below_stock_rows(self):
        primary, _ = model()
        mlx_numerics.PRIMARY.install(primary)
        self.assertEqual(mlx_rollout.prefill_batch_size(primary, mlx_rollout.Sampling(batch_size=4, prefill_step=2)), 1)
        with self.assertRaisesRegex(ValueError, "decode batch below"):
            mlx_rollout.prefill_batch_size(primary, mlx_rollout.Sampling(batch_size=SPLIT_ROWS, prefill_step=2))
        ordinary, _ = model()
        mlx_numerics.NATIVE.install(ordinary)
        self.assertEqual(mlx_rollout.prefill_batch_size(ordinary, mlx_rollout.Sampling(batch_size=4, prefill_step=2)), 4)

    def test_stock_profile_observation_carries_no_row_independent_fields(self):
        ordinary, _ = model()
        mlx_numerics.NATIVE.install(ordinary)
        observed = mlx_numerics.NATIVE.observe(ordinary)
        self.assertFalse({"projection_stock_rows", "prefill", "attention_stock_queries"} & set(observed))
        primary, _ = model()
        mlx_numerics.PRIMARY.install(primary)
        self.assertEqual({key: mlx_numerics.PRIMARY.observe(primary)[key] for key in ("projection_stock_rows", "attention_stock_queries")},
                         {"projection_stock_rows": SPLIT_ROWS, "attention_stock_queries": 64})

    def test_actual_arithmetic_is_required_by_its_assembly_binding(self):
        mx.random.seed(73)
        primary, config = model()
        mlx_numerics.PRIMARY.install(primary)
        numerical = mlx_numerics.PRIMARY.observe(primary)
        first = mlx_adapter.images(primary, config, numerics=numerical)
        with self.assertRaisesRegex(TypeError, "Actual native arithmetic"):
            mlx_numerics.NATIVE.observe(primary)
        module = primary.layers[0].linear_attn.in_proj_a
        object.__setattr__(module, "__class__", nn.QuantizedLinear)
        with self.assertRaisesRegex(TypeError, "Actual native arithmetic"):
            mlx_numerics.PRIMARY.observe(primary)
        mx.random.seed(73)
        ordinary, config = model()
        mlx_numerics.NATIVE.install(ordinary)
        second = mlx_adapter.images(ordinary, config, numerics=mlx_numerics.NATIVE.observe(ordinary))
        self.assertEqual(first["base"], second["base"])
        self.assertNotEqual(first["assembly"], second["assembly"])
        self.assertTrue(mlx_tensors.equal(mlx_adapter.state(primary), mlx_adapter.state(ordinary)))

    def test_learning_projection_scope_restores_the_actual_inference_arithmetic(self):
        mx.random.seed(71)
        inputs = mx.random.normal((ROW_COUNT, INPUT_WIDTH)).astype(mx.bfloat16)
        linear = projection()
        object.__setattr__(linear, "__class__", RowLinear)
        before = linear(inputs)
        native = nn.QuantizedLinear.__call__(linear, inputs)
        mx.eval(before, native)
        self.assertFalse(mlx_tensors.equal({"output": before}, {"output": native}))
        with mlx_training.learning(mlx_numerics.PRIMARY, linear):
            self.assertTrue(mlx_tensors.equal({"output": linear(inputs)}, {"output": native}))
        self.assertTrue(mlx_tensors.equal({"output": linear(inputs)}, {"output": before}))
        with self.assertRaisesRegex(ArithmeticError, "unfinished numerical operation"):
            with mlx_training.learning(mlx_numerics.PRIMARY, linear):
                mx.eval(linear(inputs))
                raise ArithmeticError("unfinished numerical operation")
        self.assertTrue(mlx_tensors.equal({"output": linear(inputs)}, {"output": before}))


if __name__ == "__main__":
    unittest.main()
