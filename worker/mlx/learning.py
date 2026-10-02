import math

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from worker.exchange import observation
from worker.logical import Update
from worker.mlx import probability as probability
from worker.mlx import tensors as mlx_tensors
from worker.mlx.adapter import state as adapter


def finite(values):
    mx.eval(values)
    if values.dtype != mx.float32 or not np.isfinite(np.asarray(values)).all():
        raise ValueError("Expected finite native FP32 model probabilities")
    return values


def norm(values):
    squares = [np.square(np.asarray(value, dtype=np.float64)).sum().item() for _, value in sorted(values.items())]
    result = math.sqrt(math.fsum(squares))
    if not math.isfinite(result):
        raise RuntimeError("Nonfinite native learner gradient norm")
    return result


def check_optimizer(optimizer):
    if any(not np.isfinite(np.asarray(value)).all() for _, value in tree_flatten(optimizer.state)):
        raise RuntimeError("Nonfinite native optimizer state")


def response(model, trajectory, *, evaluate, linearize):
    current, differentiate = linearize(model, trajectory, evaluate=evaluate)
    if current.dtype != mx.float32 or current.shape != trajectory.behavior.shape:
        raise ValueError("Learner graph values must be native FP32 response vectors")
    return finite(current), differentiate


def update(learner, plan, *, linearize):
    if plan.reference_source != "engine":
        raise ValueError("The MLX learner does not score reference words; declare the engine as the reference source")
    model, optimizer, exchange = learner.model, learner.optimizer, plan.exchange
    check_optimizer(optimizer)
    before = state = mlx_tensors.digest(adapter(model))
    first = set(plan.steps[0])
    later = dict.fromkeys(name for batch in plan.steps[1:] for name in batch if name not in first)
    model.train()
    proximal = {}
    for name in later:
        current = response(model, plan.trajectories[name], evaluate=learner.evaluate, linearize=linearize)[0]
        proximal[name] = probability.words(current)
        exchange.proximal(sample=name, words=proximal[name])
    currents, norms, recorded = [], None, None
    for step, batch in enumerate(plan.steps):
        parameters = adapter(model)
        accumulated = {role: {name: mx.zeros_like(value) for name, value in parameters.items()}
                       for role in ("objective", "reward")}
        consumed = []
        for name in batch:
            current, differentiate = response(model, plan.trajectories[name], evaluate=learner.evaluate, linearize=linearize)
            observed = probability.words(current)
            if step == 0:
                proximal[name] = observed
            currents.append((step, name, observed))
            objective, reward = exchange.current(step=step, sample=name, words=observed, state=state)
            cotangents = (("objective", probability.tensor(objective)), ("reward", probability.tensor(reward)))
            consumed.append(observation(tuple(word for _, value in cotangents for word in probability.words(value))))
            for role, cotangent in cotangents:
                contribution = differentiate(cotangent=cotangent)
                if contribution.keys() != parameters.keys() or any(value.dtype != mx.float32 or value.shape != parameters[key].shape
                                                                   or not np.isfinite(np.asarray(value)).all() for key, value in contribution.items()):
                    raise RuntimeError("Native differentiation produced invalid FP32 parameter gradients")
                accumulated[role] = {key: value + contribution[key] for key, value in accumulated[role].items()}
                mx.eval(accumulated[role])
            del differentiate, contribution
        measured = norm(accumulated["objective"]), norm(accumulated["reward"])
        if step == 0:
            norms = measured
            recorded = {role + "/" + key: value for role, values in accumulated.items() for key, value in values.items()}
        optimizer.update(model, tree_unflatten(list(accumulated["objective"].items())))
        mx.eval(model.trainable_parameters(), optimizer.state)
        check_optimizer(optimizer)
        after = mlx_tensors.digest(adapter(model))
        exchange.applied(step=step, before=state, after=after, consumed=consumed)
        state = after
    model.eval()
    summary = {"gradient_norm": norms[0], "reward_gradient_norm": norms[1],
               "active_tokens": sum(item.behavior.size for item in plan.trajectories.values()),
               "before": before, "after": state, "nonzero_advantages": plan.nonzero}
    return Update(summary=summary, gradients=recorded, proximal=proximal, currents=tuple(currents))
