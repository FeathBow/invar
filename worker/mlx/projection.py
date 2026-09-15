import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear

BITS = 4
GROUP_SIZE = 64
MODE = "affine"
MINIMUM_COLUMNS = 2


class RowLinear(nn.QuantizedLinear):
    def __call__(self, value):
        shape = value.shape
        result = mx.quantized_matmul(value.reshape(-1, 1, shape[-1]), self.weight[None],
                                    scales=self.scales[None], biases=self.biases[None],
                                    transpose=True, group_size=GROUP_SIZE, bits=BITS, mode=MODE)
        result = result.reshape(*shape[:-1], self.weight.shape[0])
        return result + self.bias if "bias" in self else result


class ColumnLoRALinear(LoRALinear):
    def __call__(self, value):
        base = self.linear(value)
        shape = value.shape
        columns = self.dropout(value).reshape(-1, shape[-1]).T
        count = columns.shape[-1]
        if count < MINIMUM_COLUMNS:
            # Keep single-row calls on native matrix arithmetic. The discarded
            # column has zero cotangent; no logical input is removed.
            columns = mx.repeat(columns, MINIMUM_COLUMNS, axis=-1)
        delta = self.lora_b.T @ (self.lora_a.T @ columns)
        delta = delta.T[:count].reshape(*shape[:-1], self.lora_b.shape[-1])
        return base + (self.scale * delta).astype(value.dtype)
