import math
from functools import partial

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from worker.logical import Result, ordered
from worker.mlx import probability as probability
from worker.mlx import tensors as mlx_tensors
from worker.mlx.adapter import state as adapter, install
from worker.record import loss
from worker import scalar


def evaluate_many(model, trajectories, *, evaluate):
    observed = []
    for item in trajectories:
        values = mx.stop_gradient(evaluate(model, item))
        mx.eval(values)
        if values.dtype != mx.float32 or not np.isfinite(np.asarray(values)).all():
            raise ValueError("Expected finite native FP32 model probabilities")
        observed.append(values)
    return tuple(observed)


def probabilities(model, trajectories, reference, *, evaluate):
    model.eval()
    current = adapter(model)
    proximal = evaluate_many(model, trajectories, evaluate=evaluate)
    if mlx_tensors.equal(reference, current):
        return proximal, proximal
    install(model, reference)
    fixed = evaluate_many(model, trajectories, evaluate=evaluate)
    install(model, current)
    return proximal, fixed


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


def observation(item, current, *, profile, total):
    count = item.trajectory.tokens.shape[-1] - item.trajectory.prompt_length
    roles = {"current": current, "proximal": item.proximal, "reference": item.reference,
             "behavior": item.trajectory.behavior, "advantage": mx.full(current.shape, item.advantage, dtype=mx.float32)}
    actual = probability.checked(item.trajectory.request.sample, roles, mx.ones(current.shape, dtype=mx.bool_),
                                 advantage=item.advantage, count=count)
    return probability.cotangents(actual, profile, total=total)


def norm(values):
    squares = [np.square(np.asarray(value, dtype=np.float64)).sum().item() for _, value in sorted(values.items())]
    result = math.sqrt(math.fsum(squares))
    if not math.isfinite(result):
        raise RuntimeError("Nonfinite native learner gradient norm")
    return result


def check_optimizer(optimizer):
    if any(not np.isfinite(np.asarray(value)).all() for _, value in tree_flatten(optimizer.state)):
        raise RuntimeError("Nonfinite native optimizer state")


def update(learner, batch, *, linearize):
    samples = ordered(batch)
    model, optimizer = learner.model, learner.optimizer
    check_optimizer(optimizer)
    count = sum(item.trajectory.behavior.size for item in samples)
    parameters = adapter(model)
    before = mlx_tensors.digest(parameters)
    accumulated = {role: {name: mx.zeros_like(value) for name, value in parameters.items()}
                   for role in ("objective", "reward")}
    observations = []
    model.train()
    for item in samples:
        current, differentiate = linearize(model, item.trajectory, evaluate=learner.evaluate)
        mx.eval(current)
        evaluated, objective, reward = observation(item, current, profile=batch.profile, total=count)
        observations.append(evaluated)
        for role, cotangent in (("objective", objective), ("reward", reward)):
            contribution = differentiate(cotangent=cotangent)
            if contribution.keys() != parameters.keys() or any(value.dtype != mx.float32 or value.shape != parameters[name].shape
                                                               or not np.isfinite(np.asarray(value)).all() for name, value in contribution.items()):
                raise RuntimeError("Native differentiation produced invalid FP32 parameter gradients")
            accumulated[role] = {name: value + contribution[name] for name, value in accumulated[role].items()}
            mx.eval(accumulated[role])
        # Release this trajectory's retained decoder inputs before preparing the next one.
        del differentiate, contribution
    gradient_norm, reward_norm = (norm(accumulated[role]) for role in ("objective", "reward"))
    optimizer.update(model, tree_unflatten(list(accumulated["objective"].items())))
    mx.eval(model.trainable_parameters(), optimizer.state)
    check_optimizer(optimizer)
    after = mlx_tensors.digest(adapter(model))
    model.eval()
    summary = {"loss": scalar.number(loss(observations)), "gradient_norm": gradient_norm,
               "reward_gradient_norm": reward_norm, "active_tokens": count, "before": before, "after": after,
               "nonzero_advantages": sum(item.advantage != 0 for item in samples)}
    recorded = {role + "/" + name: value for role, values in accumulated.items() for name, value in values.items()}
    return Result(summary=summary, gradients=recorded, probabilities=tuple(observations))
