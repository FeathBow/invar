import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class Profile:
    epsilon: float
    penalty: float

    def __post_init__(self):
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("Clipping epsilon must be finite and between zero and one")
        if not math.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("Reference penalty must be finite and nonnegative")


@dataclass(frozen=True, kw_only=True)
class Tokens:
    current: torch.Tensor
    proximal: torch.Tensor
    behavior: torch.Tensor
    reference: torch.Tensor
    advantage: torch.Tensor
    active: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class Reward:
    sample: str
    group: str
    value: float


def advantages(rewards, delta):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Advantage delta must be finite and positive")
    if not rewards or len({item.sample for item in rewards}) != len(rewards):
        raise ValueError("Rewards must contain distinct logical sample identities")
    groups = {}
    for item in sorted(rewards, key=lambda reward: reward.sample):
        if not item.sample or not item.group or not math.isfinite(item.value):
            raise ValueError("Reward identity and value must be defined")
        groups.setdefault(item.group, []).append(item)
    result = []
    for group in groups.values():
        if len(group) < 2:
            raise ValueError("Each logical reward group requires at least two samples")
        mean = math.fsum(item.value for item in group) / len(group)
        variance = math.fsum((item.value - mean) ** 2 for item in group) / len(group)
        denominator = math.sqrt(variance) + delta
        result.extend((item.sample, (item.value - mean) / denominator) for item in group)
    if any(not math.isfinite(value) for _, value in result):
        raise ValueError("Non-finite group advantage")
    return tuple(sorted(result))


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
