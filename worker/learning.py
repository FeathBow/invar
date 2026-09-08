from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from peft import set_peft_model_state_dict

from objective import Profile, Tokens
from objective import terms as objective_terms
from probe import adapter_state, assert_equal, digest
from probability import Observation, capture
from rollout import Trajectory


@dataclass(frozen=True, kw_only=True)
class Sample:
    trajectory: Trajectory
    proximal: torch.Tensor
    reference: torch.Tensor
    advantage: float


@dataclass(frozen=True, kw_only=True)
class Batch:
    samples: tuple[Sample, ...]
    order: tuple[str, ...]
    profile: Profile


@dataclass(frozen=True, kw_only=True)
class Learner:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    evaluate: Callable[[torch.nn.Module, Trajectory], torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class Result:
    summary: dict
    gradients: dict[str, torch.Tensor]
    probabilities: tuple[Observation, ...]


def ordered(batch):
    if not batch.order or any(not name for name in batch.order) or len(set(batch.order)) != len(batch.order):
        raise ValueError("Logical batch order must contain distinct sample identities")
    samples = {item.trajectory.request.sample: item for item in batch.samples}
    if len(samples) != len(batch.samples) or set(samples) != set(batch.order):
        raise ValueError("Delivered samples do not match the declared logical batch")
    return tuple(samples[name] for name in batch.order)


def probabilities(model, trajectories, reference, *, evaluate):
    current = adapter_state(model)
    with torch.no_grad():
        proximal = tuple(evaluate(model, item).cpu() for item in trajectories)
        set_peft_model_state_dict(model, reference)
        assert_equal(reference, adapter_state(model))
        fixed = tuple(evaluate(model, item).cpu() for item in trajectories)
        set_peft_model_state_dict(model, current)
        assert_equal(current, adapter_state(model))
    return proximal, fixed


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


def update(learner, batch):
    samples = ordered(batch)
    model, optimizer = learner.model, learner.optimizer
    check_optimizer(optimizer)
    count = sum(item.trajectory.behavior.numel() for item in samples)
    before = digest(adapter_state(model))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    observations = []
    trainable = parameters(model)
    reward_gradients = tuple(torch.zeros_like(value) for value in trainable)
    for item in samples:
        current = learner.evaluate(model, item.trajectory)
        tokens = Tokens(current=current, proximal=item.proximal.to(current.device),
                        behavior=item.trajectory.behavior.to(current.device),
                        reference=item.reference.to(current.device),
                        advantage=torch.full_like(current, item.advantage),
                        active=torch.ones_like(current, dtype=torch.bool))
        value = objective_terms(tokens, batch.profile).sum() / count
        observations.append(capture(item.trajectory.request.sample, tokens))
        reward_loss = objective_terms(tokens, replace(batch.profile, penalty=0)).sum() / count
        contribution = torch.autograd.grad(reward_loss, trainable, retain_graph=True)
        reward_gradients = tuple(total + addition for total, addition in zip(reward_gradients, contribution))
        value.backward()
        losses.append(value.detach().item())
    norm = gradient_norm(tuple(value.grad for value in trainable))
    reward_norm = gradient_norm(reward_gradients)
    recorded = capture_gradients(model, reward_gradients)
    optimizer.step()
    check_optimizer(optimizer)
    if any(not value.isfinite().all() for value in parameters(model)):
        raise RuntimeError("Non-finite learner parameter after update")
    model.eval()
    summary = {"loss": sum(losses), "gradient_norm": norm, "reward_gradient_norm": reward_norm, "active_tokens": count,
               "before": before, "after": digest(adapter_state(model)),
               "nonzero_advantages": sum(item.advantage != 0 for item in samples)}
    return Result(summary=summary, gradients=recorded, probabilities=tuple(observations))
