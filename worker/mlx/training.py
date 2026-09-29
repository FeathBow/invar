from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

from worker.mlx.numerics import inventory, qualified
from worker.mlx.temperature import tempered

LINEAR = {"independent-native-rows/v8": nn.QuantizedLinear}


@contextmanager
def learning(profile, model):
    selected = LINEAR.get(profile.name)
    if selected is None:
        yield
        return
    modules = tuple(inventory(model, nn.QuantizedLinear).values())
    if not modules or any(type(module) is not profile.linear for module in modules):
        raise TypeError("Native learning requires its declared inference projection inventory")
    try:
        for module in modules:
            object.__setattr__(module, "__class__", selected)
        yield
        if any(type(module) is not selected for module in modules):
            raise TypeError("Actual native learning projections changed inside their owned operation")
    finally:
        for module in modules:
            object.__setattr__(module, "__class__", profile.linear)


def description(profile):
    selected = LINEAR.get(profile.name)
    projection = {} if selected is None else {
        "learning_projection": {"module": qualified(selected),
                                "schedule": "one complete logical trajectory per native projection",
                                "roles": ["objective_vjp", "reward_vjp"],
                                "lifetime": "owned numerical operation; inference classes restored before state observation"}}
    return {"probabilities": {"behavior": "engine words of the request policy's own rollout",
                              "reference": "engine forced-path scoring inside the rollout transaction",
                              "proximal": "learner graph at the rollout temperature before the first optimizer step",
                              "current": "learner graph at the rollout temperature at the weights of each optimizer step",
                              "temperature": "rollout request"},
            **projection}


def logprobs(model, trajectory):
    return selected_logprobs(model(trajectory.tokens[:, :-1]), trajectory)


def selected_logprobs(logits, trajectory):
    tokens = trajectory.tokens
    response = tokens[:, trajectory.prompt_length:]
    logits = logits[:, trajectory.prompt_length - 1:, :].astype(mx.float32)
    distribution = tempered(logits - mx.logsumexp(logits, axis=-1, keepdims=True), trajectory.request.temperature)[0]
    return mx.log(mx.take_along_axis(distribution, response[:, :, None], axis=-1)).reshape(-1)
