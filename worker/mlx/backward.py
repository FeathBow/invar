from functools import partial

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from worker.mlx.adapter import state as adapter
from worker.mlx.words import words
from worker.mlx.rollout import selected_logprobs
from worker.mlx import tensors as mlx_tensors
from worker.mlx import recurrentvjp as mlx_recurrent_backward


def checkpoints(model, trajectory):
    hidden = model.language_model.model.embed_tokens(trajectory.tokens[:, :-1])
    mx.eval(hidden)
    masks = (create_attention_mask(hidden, None), create_ssm_mask(hidden, None))
    states = [hidden]
    for index, layer in enumerate(model.layers):
        hidden = transform(layer, hidden, mask=masks[int(layer.is_linear)], cotangent=mx.zeros_like(hidden),
                           input_gradient=index != 0)[0][0]
        mx.eval(hidden)
        states.append(hidden)
    return states, masks


def head(model, hidden, trajectory):
    language = model.language_model
    normalized = language.model.norm(hidden)
    logits = language.model.embed_tokens.as_linear(normalized) if language.args.tie_word_embeddings else language.lm_head(normalized)
    return selected_logprobs(logits, trajectory)


def transform(layer, hidden, *, mask, cotangent, input_gradient=True):
    original = dict(tree_flatten(layer.trainable_parameters()))
    names = tuple(sorted(original))

    def forward(*arguments):
        inputs, *parameters = arguments if input_gradient else (hidden, *arguments)
        layer.update(tree_unflatten(list(zip(names, parameters, strict=True))))
        return layer(inputs, mask=mask, cache=None)

    parameters = [original[name] for name in names]
    try:
        return mx.vjp(forward, [hidden, *parameters] if input_gradient else parameters, [cotangent])
    finally:
        layer.update(tree_unflatten(list(original.items())))


def layer_vjp(layer, hidden, *, mask, cotangent, expected, input_gradient=True):
    if layer.is_linear and input_gradient:
        return mlx_recurrent_backward.vjp(layer, hidden, mask=mask, cotangent=cotangent, expected=expected)
    output, gradients = transform(layer, hidden, mask=mask, cotangent=cotangent, input_gradient=input_gradient)
    mx.eval(output, gradients)
    if len(output) != 1 or not mlx_tensors.equal({"hidden": output[0]}, {"hidden": expected}):
        raise RuntimeError("Native layer recomputation differs from its retained forward input chain")
    names = sorted(dict(tree_flatten(layer.trainable_parameters())))
    return (gradients[0] if input_gradient else None), dict(zip(names, gradients[1:] if input_gradient else gradients, strict=True))


def linearize(model, trajectory, *, evaluate):
    states, masks = checkpoints(model, trajectory)
    current = mx.vjp(lambda value: head(model, value, trajectory), [states[-1]],
                     [mx.zeros((trajectory.tokens.shape[-1] - trajectory.prompt_length,), dtype=mx.float32)])[0][0]
    mx.eval(current)
    return current, partial(vjp, model, trajectory, states=states, masks=masks, current=current)


def vjp(model, trajectory, *, states, masks, current, cotangent):
    output, hidden = mx.vjp(lambda value: head(model, value, trajectory), [states[-1]], [cotangent])
    mx.eval(output, hidden)
    if len(output) != 1 or words(output[0]) != words(current):
        raise RuntimeError("Native layerwise differentiation differs from the checked model probabilities")
    names = {id(module): name for name, module in model.named_modules()}
    carried, = hidden
    collected = {}
    for index in reversed(range(len(model.layers))):
        layer = model.layers[index]
        carried, gradients = layer_vjp(layer, states[index], mask=masks[int(layer.is_linear)],
                                       cotangent=carried, expected=states[index + 1], input_gradient=index != 0)
        collected.update({names[id(layer)] + "." + name: value for name, value in gradients.items()})
    parameters = adapter(model)
    if collected.keys() != parameters.keys():
        raise RuntimeError("Native layerwise gradients omit or add trainable model parameters")
    return collected
