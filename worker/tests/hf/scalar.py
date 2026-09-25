import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import math
import unittest
from dataclasses import replace

import torch

from worker import scalar
from worker.hf.objective import Profile, Tokens, terms
from worker.hf.probability import checked, checked_cotangents, cotangents, words

NEGATIVE_ZERO = 1 << 31
LARGE = 2 ** 24
ANALYTIC_TOLERANCE = 2e-6


def inputs(current, proximal, advantage):
    return scalar.Inputs(current=scalar.word(current), proximal=scalar.word(proximal),
                         behavior=scalar.word(proximal), reference=scalar.word(current),
                         advantage=scalar.word(advantage))


class ScalarTests(unittest.TestCase):
    def test_equal_roles_and_real_tensor_vjp_have_the_analytic_result(self):
        current = torch.tensor([-1.0, -1.0], requires_grad=True)
        tokens = Tokens(current=current, proximal=current, behavior=current, reference=current,
                        advantage=torch.full_like(current, 2), active=torch.ones(2, dtype=torch.bool))
        observed = checked("sample", tokens, advantage=2, count=2)
        evaluated, gradient, reward = cotangents(observed, Profile(epsilon=0.2, penalty=0.125),
                                                 total=2, device=current.device, linearized=current)
        self.assertEqual(evaluated.objective, (scalar.Output(term=scalar.word(-2),
                         gradient=scalar.word(-1), reward_gradient=scalar.word(-1)),) * 2)
        current.backward(gradient)
        self.assertTrue(torch.equal(current.grad, torch.tensor([-1.0, -1.0])))
        self.assertEqual(words(reward), words(gradient))

    def test_clipped_branches_and_rounded_minimum_ties(self):
        profile = Profile(epsilon=0.5, penalty=0)
        cases = ((-math.log(4), 0, 1, -0.25, -0.25), (-math.log(2), 0, 1, -0.5, -0.5),
                 (0, 0, 1, -1, -1), (0, -math.log(1.5), 1, -1.5, -1.5),
                 (0, -math.log(2), 1, -1.5, 0), (-math.log(4), 0, -1, 0.5, 0),
                 (-math.log(2), 0, -1, 0.5, 0.5), (0, 0, -1, 1, 1),
                 (0, -math.log(1.5), -1, 1.5, 1.5), (0, -math.log(2), -1, 2, 2))
        for current, proximal, advantage, term, gradient in cases:
            with self.subTest(current=current, proximal=proximal, advantage=advantage):
                result, = scalar.calculate(profile, 1, (inputs(current, proximal, advantage),))
                self.assertEqual(result.term, scalar.word(term))
                self.assertEqual(result.gradient, scalar.word(gradient))

    def test_ordinary_values_match_the_independent_fp64_formula_and_autograd(self):
        profile = Profile(epsilon=0.5, penalty=0.125)
        probabilities = ((0.4, 0.3, 0.2, 0.2, 2), (0.1, 0.4, 0.2, 0.3, -1),
                         (0.8, 0.4, 0.5, 0.6, 1))
        values = tuple(scalar.Inputs(**dict(zip(("current", "proximal", "behavior", "reference", "advantage"),
                        [*(scalar.word(math.log(value)) for value in row[:4]), scalar.word(row[4])], strict=True)))
                       for row in probabilities)
        columns = {name: torch.tensor([scalar.number(getattr(value, name)) for value in values],
                                      dtype=torch.float64, requires_grad=True)
                   for name in scalar.Inputs.__dataclass_fields__}
        tokens = Tokens(**columns, active=torch.ones(len(values), dtype=torch.bool))
        independent = terms(tokens, profile)
        independent.mean().backward()
        result = scalar.calculate(profile, len(values), values)
        for item, value, gradient in zip(result, independent.tolist(), tokens.current.grad.tolist(), strict=True):
            self.assertAlmostEqual(scalar.number(item.term), value, delta=ANALYTIC_TOLERANCE)
            self.assertAlmostEqual(scalar.number(item.gradient), gradient, delta=ANALYTIC_TOLERANCE)
        for name in ("proximal", "behavior", "reference", "advantage"):
            self.assertIsNone(columns[name].grad)

    def test_logical_sum_rounds_each_addition_and_divides_only_once(self):
        first = tuple(map(scalar.word, (LARGE, 1, -LARGE)))
        second = tuple(map(scalar.word, (LARGE, -LARGE, 1)))
        self.assertEqual(scalar.mean32(first), scalar.word(0))
        self.assertEqual(scalar.mean32(second), scalar.word(1 / 3))
        self.assertEqual(scalar.mean32(tuple(map(scalar.word, (1, 2, 3)))), scalar.word(2))
        self.assertNotEqual(scalar.mean32(first), scalar.word(math.fsum(map(scalar.number, first)) / 3))

    def test_actual_cotangent_words_dtype_and_shape_are_checked(self):
        profile = Profile(epsilon=0.2, penalty=0)
        result, = scalar.calculate(profile, 1, (inputs(0, 0, 0),))
        expected = ((result.gradient,), (result.reward_gradient,))
        objective = torch.tensor([0.0])
        reward = torch.tensor([-0.0])
        self.assertEqual(expected, ((0,), (NEGATIVE_ZERO,)))
        self.assertEqual(checked_cotangents(expected, (objective, reward), device=objective.device), expected)
        for changed in ((objective.double(), reward), (objective, reward.reshape(())),
                        (objective, -reward), (objective + 1, reward)):
            with self.subTest(tensors=changed), self.assertRaises(ValueError):
                checked_cotangents(expected, changed, device=objective.device)

    def test_nonfinite_ratios_conversion_and_sum_fail_explicitly(self):
        profile = Profile(epsilon=0.2, penalty=0.04)
        ordinary = inputs(-1, -1, 1)
        for value in (replace(ordinary, current=scalar.word(1)),
                      replace(ordinary, current=scalar.word(-1000)),
                      replace(ordinary, proximal=scalar.word(-1000)),
                      replace(ordinary, reference=scalar.word(-1000)),
                      replace(ordinary, advantage=0x7F800000)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scalar.calculate(profile, 1, (value,))
        with self.assertRaises(ValueError):
            scalar.calculate(Profile(epsilon=0.2, penalty=1e100), 1, (ordinary,))
        for values in ((), (0x7FC00000,), (scalar.word(3e38),) * 2):
            with self.subTest(values=values), self.assertRaises(ValueError):
                scalar.mean32(values)


if __name__ == "__main__":
    unittest.main()
