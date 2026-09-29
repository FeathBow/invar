from functools import partial

import mlx.core as mx
from mlx.utils import tree_unflatten
import numpy as np

from worker.mlx import probability
from worker.mlx.adapter import state as adapter


def transform(model, trajectory, *, evaluate, cotangent):
    original = adapter(model)
    names = tuple(sorted(original))

    def forward(*parameters):
        model.update(tree_unflatten(list(zip(names, parameters, strict=True))))
        return evaluate(model, trajectory)

    try:
        outputs, gradients = mx.vjp(forward, [original[name] for name in names], [cotangent])
    finally:
        model.update(tree_unflatten(list(original.items())))
    return outputs, dict(zip(names, gradients, strict=True))


def vjp(model, trajectory, *, evaluate, current, cotangent):
    original = adapter(model)
    outputs, observed = transform(model, trajectory, evaluate=evaluate, cotangent=cotangent)
    mx.eval(outputs, observed)
    if len(outputs) != 1 or probability.words(outputs[0]) != probability.words(current):
        raise RuntimeError("Native differentiation probabilities differ from the checked objective input")
    if any(value.dtype != mx.float32 or value.shape != original[name].shape
           or not np.isfinite(np.asarray(value)).all() for name, value in observed.items()):
        raise RuntimeError("Native model produced invalid FP32 adapter gradients")
    return observed


def linearize(model, trajectory, *, evaluate):
    seed = mx.zeros((trajectory.tokens.shape[-1] - trajectory.prompt_length,), dtype=mx.float32)
    current = transform(model, trajectory, evaluate=evaluate, cotangent=seed)[0][0]
    mx.eval(current)
    return current, partial(vjp, model, trajectory, evaluate=evaluate, current=current)
