from contextlib import contextmanager

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import gated_delta
from mlx_lm.models.qwen3_5 import GatedDeltaNet

from worker.mlx import recurrence as mlx_recurrence
from worker.mlx import tensors as mlx_tensors


@contextmanager
def boundary(layer, operation):
    module = layer.linear_attn
    original = type(module)
    update = mlx_recurrence.bind(gated_delta.gated_delta_update, name="gated_delta_ops", operation=operation)
    call = mlx_recurrence.bind(GatedDeltaNet.__call__, name="gated_delta_update", operation=update)
    selected = type("RecurrentBoundary", (original,), {"__call__": call})
    object.__setattr__(module, "__class__", selected)
    try:
        yield
    finally:
        object.__setattr__(module, "__class__", original)


def prefix(layer, hidden, *, mask):
    captured = []

    def observe(q, k, v, g, beta, state, mask=None):
        captured.append((q, k, v, g, beta, state))
        return mlx_recurrence.segmented(q, k, v, g, beta, state, mask)

    with boundary(layer, observe):
        layer.linear_attn(hidden, mask=mask, cache=None)
    values, = captured
    return values


def segments(inputs, *, mask):
    outputs, states = [], [inputs[-1]]
    count = mlx_recurrence.CHECKPOINT_TOKENS
    for start in range(0, inputs[0].shape[1], count):
        interval = slice(start, start + count)
        values = [value[:, interval] for value in inputs[:-1]]
        output, state = gated_delta.gated_delta_ops(*values, states[-1], mask=None if mask is None else mask[:, interval])
        mx.eval(output, state)
        outputs.append(output)
        states.append(state)
    return outputs, states


def recurrence_vjp(inputs, outputs, states, *, cotangent, mask):
    carried = mx.zeros_like(states[-1])
    collected = [[] for _ in inputs[:-1]]
    count = mlx_recurrence.CHECKPOINT_TOKENS
    for index in reversed(range(len(outputs))):
        interval = slice(index * count, (index + 1) * count)
        values = [*[value[:, interval] for value in inputs[:-1]], states[index]]
        selected = None if mask is None else mask[:, interval]
        _, gradients = mx.vjp(lambda *values: gated_delta.gated_delta_ops(*values, mask=selected),
                              values, [cotangent[:, interval], carried])
        mx.eval(gradients)
        for destination, gradient in zip(collected, gradients[:-1], strict=True):
            destination.append(gradient)
        carried = gradients[-1]
    gradients = tuple(mx.concatenate(list(reversed(values)), axis=1) for values in collected)
    mx.eval(gradients, carried)
    return (*gradients, carried)


@contextmanager
def normalization(layer, value):
    module = layer.input_layernorm
    original = type(module)
    selected = type("NormalizedBoundary", (original,), {"__call__": lambda _self, _inputs: value})
    object.__setattr__(module, "__class__", selected)
    try:
        yield
    finally:
        object.__setattr__(module, "__class__", original)


def suffix_vjp(layer, hidden, recurrent, *, normalized, state, cotangent, mask, expected):
    original = dict(tree_flatten(layer.trainable_parameters()))
    names = tuple(sorted(original))

    def forward(inputs, normed, response, *parameters):
        layer.update(tree_unflatten(list(zip(names, parameters, strict=True))))
        with normalization(layer, normed), boundary(layer, lambda *_values: (response, state)):
            return layer(inputs, mask=mask, cache=None)

    try:
        output, gradients = mx.vjp(forward, [hidden, normalized, recurrent, *[original[name] for name in names]], [cotangent])
    finally:
        layer.update(tree_unflatten(list(original.items())))
    mx.eval(output, gradients)
    if not mlx_tensors.equal({"hidden": output[0]}, {"hidden": expected}):
        raise RuntimeError("Native recurrent decoder suffix differs from its retained VJP primal")
    return gradients[0], gradients[1], gradients[2], dict(zip(names, gradients[3:], strict=True))


def vjp(layer, hidden, *, mask, cotangent, expected):
    normalized = layer.input_layernorm(hidden)
    mx.eval(normalized)
    operation = lambda value: prefix(layer, value, mask=mask)
    shapes = operation(normalized)
    inputs = mx.vjp(operation, [normalized], [mx.zeros_like(value) for value in shapes])[0]
    mx.eval(inputs)
    outputs, states = segments(inputs, mask=mask)
    recurrent = mx.concatenate(outputs, axis=1)
    mx.eval(recurrent)
    direct, gated, incoming, parameters = suffix_vjp(layer, hidden, recurrent, normalized=normalized,
                                                     state=states[-1], cotangent=cotangent, mask=mask, expected=expected)
    cotangents = recurrence_vjp(inputs, outputs, states, cotangent=incoming, mask=mask)
    observed, gradients = mx.vjp(operation, [normalized], list(cotangents))
    mx.eval(observed, gradients)
    if not mlx_tensors.equal(dict(enumerate(inputs)), dict(enumerate(observed))):
        raise RuntimeError("Native recurrent prefix differs between primal and derivative")
    _, normalized_gradient = mx.vjp(layer.input_layernorm, [hidden], [gradients[0] + gated])
    carried = direct + normalized_gradient[0]
    mx.eval(carried)
    return carried, parameters
