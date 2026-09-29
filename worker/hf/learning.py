import torch

from worker.exchange import observation
from worker.hf.model import adapter_state
from worker.hf.tensors import digest
from worker.hf.probability import tensor, words
from worker.logical import Update


def parameters(model):
    return tuple(value for value in model.parameters() if value.requires_grad)


def gradient_norm(gradients):
    if not gradients or any(value is None or not value.isfinite().all() for value in gradients):
        raise RuntimeError("Missing or non-finite learner gradients")
    return torch.stack([value.double().square().sum() for value in gradients]).sum().sqrt().item()


def check_optimizer(optimizer):
    if any(not slot.isfinite().all() for slots in optimizer.state.values() for slot in slots.values()):
        raise RuntimeError("Non-finite optimizer state")


def capture_gradients(model, reward_gradients):
    named = tuple((name, value) for name, value in model.named_parameters() if value.requires_grad)
    recorded = {}
    for (name, parameter), reward in zip(named, reward_gradients, strict=True):
        recorded[f"objective/{name}"] = parameter.grad.detach().cpu().contiguous().clone()
        recorded[f"reward/{name}"] = reward.detach().cpu().contiguous().clone()
    return recorded


def parameter_vjps(current, trainable, cotangents):
    # One batched VJP applies both cotangent rows to the same forward graph.
    gradients = torch.autograd.grad(current, trainable, grad_outputs=cotangents, is_grads_batched=True)
    objective_values, reward_values = zip(*(value.unbind() for value in gradients), strict=True)
    return objective_values, reward_values


def accumulate_objective(trainable, contribution):
    for parameter, gradient in zip(trainable, contribution, strict=True):
        parameter.grad = gradient.clone() if parameter.grad is None else parameter.grad + gradient


def response(learner, trajectory):
    value = learner.evaluate(learner.model, trajectory)
    if value.dtype != torch.float32 or value.shape != trajectory.behavior.shape:
        raise ValueError("Learner graph values must be FP32 response vectors")
    return value


def update(learner, plan):
    model, optimizer, exchange = learner.model, learner.optimizer, plan.exchange
    check_optimizer(optimizer)
    before = state = digest(adapter_state(model))
    first = set(plan.steps[0])
    later = dict.fromkeys(name for batch in plan.steps[1:] for name in batch if name not in first)
    model.train()
    proximal = {}
    with torch.no_grad():
        for name in later:
            proximal[name] = words(response(learner, plan.trajectories[name]))
            exchange.proximal(sample=name, words=proximal[name])
    trainable = parameters(model)
    currents, norms, recorded = [], None, None
    for step, batch in enumerate(plan.steps):
        optimizer.zero_grad(set_to_none=True)
        reward_gradients = tuple(torch.zeros_like(value) for value in trainable)
        consumed = []
        for name in batch:
            linearized = response(learner, plan.trajectories[name])
            observed = words(linearized)
            if step == 0:
                proximal[name] = observed
            currents.append((step, name, observed))
            objective, reward = exchange.current(step=step, sample=name, words=observed, state=state)
            cotangents = tensor(objective + reward, device=linearized.device).reshape(2, -1)
            consumed.append(observation(words(cotangents.reshape(-1))))
            objective_values, reward_values = parameter_vjps(linearized, trainable, cotangents)
            reward_gradients = tuple(total + addition for total, addition in zip(reward_gradients, reward_values, strict=True))
            accumulate_objective(trainable, objective_values)
        measured = gradient_norm(tuple(value.grad for value in trainable)), gradient_norm(reward_gradients)
        if step == 0:
            norms, recorded = measured, capture_gradients(model, reward_gradients)
        optimizer.step()
        check_optimizer(optimizer)
        if any(not value.isfinite().all() for value in parameters(model)):
            raise RuntimeError("Non-finite learner parameter after update")
        after = digest(adapter_state(model))
        exchange.applied(step=step, before=state, after=after, consumed=consumed)
        state = after
    model.eval()
    summary = {"gradient_norm": norms[0], "reward_gradient_norm": norms[1],
               "active_tokens": sum(value.behavior.numel() for value in plan.trajectories.values()),
               "before": before, "after": state, "nonzero_advantages": plan.nonzero}
    return Update(summary=summary, gradients=recorded, proximal=proximal, currents=tuple(currents))
