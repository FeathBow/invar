from types import FunctionType

import mlx.core as mx
from mlx_lm.models import gated_delta
from mlx_lm.models.qwen3_5 import GatedDeltaNet

CHECKPOINT_TOKENS = 16


def segmented(q, k, v, g, beta, state=None, mask=None):
    if state is None:
        state = mx.zeros((q.shape[0], v.shape[-2], v.shape[-1], q.shape[-1]), dtype=mx.float32)
    outputs = []
    operation = mx.checkpoint(gated_delta.gated_delta_ops)
    for start in range(0, q.shape[1], CHECKPOINT_TOKENS):
        interval = slice(start, start + CHECKPOINT_TOKENS)
        selected = (value[:, interval] for value in (q, k, v, g, beta))
        output, state = operation(*selected, state, None if mask is None else mask[:, interval])
        outputs.append(output)
    return mx.concatenate(outputs, axis=1), state


def bind(function, *, name, operation):
    if name not in function.__code__.co_names:
        raise ValueError("The pinned native call no longer exposes its declared recurrent operation")
    namespace = {**function.__globals__, name: operation}
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


update = bind(gated_delta.gated_delta_update, name="gated_delta_ops", operation=segmented)


class CheckpointedDeltaNet(GatedDeltaNet):
    __call__ = bind(GatedDeltaNet.__call__, name="gated_delta_update", operation=update)


def install(model):
    for layer in model.layers:
        if layer.is_linear:
            module = layer.linear_attn
            if type(module) is not GatedDeltaNet:
                raise ValueError("Native recurrent checkpointing requires the pinned GatedDeltaNet module")
            object.__setattr__(module, "__class__", CheckpointedDeltaNet)


def verify(model):
    if any(type(layer.linear_attn) is not CheckpointedDeltaNet for layer in model.layers if layer.is_linear):
        raise ValueError("Native model differs from its declared recurrent checkpoint implementation")
