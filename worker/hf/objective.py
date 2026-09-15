from dataclasses import dataclass

import torch

from worker.scalar import Profile
from worker.advantage import Reward, advantages


@dataclass(frozen=True, kw_only=True)
class Tokens:
    current: torch.Tensor
    proximal: torch.Tensor
    behavior: torch.Tensor
    reference: torch.Tensor
    advantage: torch.Tensor
    active: torch.Tensor


def active_inputs(tokens):
    fields = (tokens.current, tokens.proximal, tokens.behavior,
              tokens.reference, tokens.advantage)
    if tokens.active.dtype != torch.bool or tokens.active.ndim != 1:
        raise ValueError("Active-token mask must be a Boolean vector")
    if tokens.current.dtype not in (torch.float32, torch.float64):
        raise ValueError("Objective arithmetic requires an explicit FP32 or FP64 profile")
    for value in fields:
        if value.shape != tokens.active.shape or value.device != tokens.active.device:
            raise ValueError("Probability roles and mask must have identical shape and device")
        if value.dtype != tokens.current.dtype:
            raise ValueError("Probability roles and advantages must use the objective dtype")
    if not tokens.active.any():
        raise ValueError("An update requires at least one active response token")
    selected = tuple(value[tokens.active] for value in fields)
    if any(not value.isfinite().all() for value in selected):
        raise ValueError("Non-finite active objective input")
    if any((value > 0).any() for value in selected[:4]):
        raise ValueError("Log probabilities must be nonpositive")
    current, *fixed = selected
    return (current, *(value.detach() for value in fixed))


def surrogate(ratio, advantage, epsilon):
    lower, upper = 1 - epsilon, 1 + epsilon
    clipped = torch.where(ratio <= lower, lower, torch.where(ratio >= upper, upper, ratio))
    direct, bounded = ratio * advantage, clipped * advantage
    return torch.where(direct <= bounded, direct, bounded)


def terms(tokens, profile):
    current, proximal, behavior, reference, advantage = active_inputs(tokens)
    weight = (proximal - behavior).exp()
    ratio = (current - proximal).exp()
    reference_ratio = (reference - current).exp()
    for value in (weight, ratio, reference_ratio):
        if not value.isfinite().all() or (value <= 0).any():
            raise ValueError("Probability ratio overflow or underflow")
    penalty = reference_ratio - (reference - current) - 1
    result = -weight * surrogate(ratio, advantage, profile.epsilon) + profile.penalty * penalty
    if not result.isfinite().all():
        raise ValueError("Non-finite objective result")
    return result
