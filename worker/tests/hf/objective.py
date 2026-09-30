import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import math
import unittest
from dataclasses import dataclass, replace

import torch

from worker.tests.advantage import Reward, advantages
from worker.tests.scalar import Profile

DTYPE = torch.float64
EPSILON = 0.5
PENALTY = 0.125
TOLERANCE = 1e-12


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


def tensor(values):
    return torch.tensor(values, dtype=DTYPE, requires_grad=True)


def inputs():
    return Tokens(current=tensor([math.log(0.4), math.log(0.1), math.log(0.8)]),
                  proximal=tensor([math.log(0.3), math.log(0.4), math.log(0.4)]),
                  behavior=tensor([math.log(0.2), math.log(0.2), math.log(0.5)]),
                  reference=tensor([math.log(0.2), math.log(0.3), math.log(0.6)]),
                  advantage=tensor([2, -1, 1]), active=torch.tensor([True, True, True]))


def expected(values, profile):
    current, proximal, behavior, reference, advantage = values
    p, old, b, q = map(math.exp, (current, proximal, behavior, reference))
    ratio = p / old
    clipped = min(1 + profile.epsilon, max(1 - profile.epsilon, ratio))
    loss = -old / b * min(ratio * advantage, clipped * advantage)
    loss += profile.penalty * (q / p - math.log(q / p) - 1)
    selected = (advantage > 0 and ratio <= 1 + profile.epsilon)
    selected |= advantage < 0 and ratio >= 1 - profile.epsilon
    gradient = -advantage * p / b * selected + profile.penalty * (1 - q / p)
    return loss, gradient


class ObjectiveTests(unittest.TestCase):
    def assert_values(self, actual, wanted):
        self.assertEqual(len(actual), len(wanted))
        for value, reference in zip(actual, wanted):
            self.assertAlmostEqual(value, reference, delta=TOLERANCE)

    def test_formula_and_only_current_differentiates(self):
        batch = inputs()
        profile = Profile(epsilon=EPSILON, penalty=PENALTY)
        columns = (batch.current, batch.proximal, batch.behavior, batch.reference, batch.advantage)
        reference = [expected(values, profile) for values in zip(*(x.tolist() for x in columns))]
        result = terms(batch, profile)
        self.assert_values(result.tolist(), [loss for loss, _ in reference])
        result.mean().backward()
        self.assert_values(batch.current.grad.tolist(), [gradient / len(reference) for _, gradient in reference])
        for field in columns[1:]:
            self.assertIsNone(field.grad)

    def test_boundary_and_tie_selection(self):
        ratio = tensor([0.25, 0.5, 1, 1.5, 2, 0.25, 0.5, 1, 1.5, 2])
        advantage = torch.tensor([1] * 5 + [-1] * 5, dtype=DTYPE)
        result = surrogate(ratio, advantage, EPSILON)
        result.sum().backward()
        self.assert_values(result.tolist(), [0.25, 0.5, 1, 1.5, 1.5, -0.5, -0.5, -1, -1.5, -2])
        self.assert_values(ratio.grad.tolist(), [1, 1, 1, 1, 0, 0, -1, -1, -1, -1])

    def test_masked_invalid_roles_are_not_evaluated(self):
        batch = inputs()
        fields = {name: tensor([getattr(batch, name)[0].item(), float("nan")])
                  for name in ("current", "proximal", "behavior", "reference", "advantage")}
        selected = Tokens(**fields, active=torch.tensor([True, False]))
        profile = Profile(epsilon=EPSILON, penalty=PENALTY)
        result = terms(selected, profile)
        result.sum().backward()
        self.assertEqual(result.numel(), 1)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(selected.current.grad[1].item(), 0)

    def test_invalid_inputs_fail(self):
        batch = inputs()
        profile = Profile(epsilon=EPSILON, penalty=PENALTY)
        bad = (replace(batch, active=torch.zeros(3, dtype=torch.bool)),
               replace(batch, behavior=tensor([0])),
               replace(batch, current=tensor([float("nan"), -1, -1])),
               replace(batch, current=tensor([1, -1, -1])),
               replace(batch, current=tensor([-10000, -1, -1])),
               replace(batch, active=torch.ones(3)),
               replace(batch, reference=batch.reference.float()))
        for candidate in bad:
            with self.assertRaises(ValueError):
                terms(candidate, profile)

    def test_logical_groups_not_delivery_order(self):
        rewards = (Reward(sample="a", group="first", value=0),
                   Reward(sample="b", group="first", value=1),
                   Reward(sample="c", group="second", value=0),
                   Reward(sample="d", group="second", value=0))
        delta = 0.25
        correct = (("a", -2 / 3), ("b", 2 / 3), ("c", 0), ("d", 0))
        self.assertEqual(advantages(rewards, delta), correct)
        reordered = tuple(rewards[index] for index in (3, 1, 0, 2))
        self.assertEqual(advantages(reordered, delta), correct)

    def test_invalid_groups_and_profile(self):
        sample = Reward(sample="a", group="first", value=0)
        for values in ((), (sample,), (sample, sample),
                       (sample, replace(sample, sample="b", value=float("nan")))):
            with self.assertRaises(ValueError):
                advantages(values, 0.25)
        with self.assertRaises(ValueError):
            advantages((sample, replace(sample, sample="b")), 0)
        for epsilon, penalty in ((0, 0), (1, 0), (0.5, -1), (float("nan"), 0)):
            with self.assertRaises(ValueError):
                Profile(epsilon=epsilon, penalty=penalty)

    def test_chain_rule_reaches_model_parameters(self):
        weight = tensor([0.2, -0.4])
        features = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=DTYPE)
        selected = torch.log_softmax(features @ weight, dim=0)[:1]
        batch = Tokens(current=selected, proximal=tensor([math.log(0.5)]),
                       behavior=tensor([math.log(0.4)]), reference=tensor([math.log(0.3)]),
                       advantage=tensor([1]), active=torch.tensor([True]))
        profile = Profile(epsilon=EPSILON, penalty=PENALTY)
        terms(batch, profile).sum().backward()
        probability = selected.exp().item()
        _, scalar = expected((selected.item(), math.log(0.5), math.log(0.4), math.log(0.3), 1), profile)
        wanted = [scalar * (1 - probability) * value for value in (2, 1.5)]
        self.assert_values(weight.grad.tolist(), wanted)
        self.assertGreater(weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
