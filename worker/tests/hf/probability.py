import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from dataclasses import replace
from itertools import permutations

import torch

from worker.hf.learning import update
from worker.tests.hf.learning import batch, make_learner
from worker.hf.objective import Tokens
from worker.hf.probability import ROLES, capture, checked, words

NEGATIVE_ZERO = 0x80000000


class ProbabilityTests(unittest.TestCase):
    def test_actual_update_records_the_pre_step_values_in_logical_order(self):
        logical = batch()
        logical = replace(logical, samples=tuple(replace(item, advantage=1) for item in logical.samples))
        learner = make_learner()
        before = tuple(words(learner.evaluate(learner.model, item.trajectory)) for item in logical.samples)
        result = update(learner, logical)
        self.assertEqual(tuple(item.sample for item in result.probabilities), logical.order)
        self.assertEqual(tuple(item.words[ROLES.index("current")] for item in result.probabilities), before)
        after = tuple(words(learner.evaluate(learner.model, item.trajectory)) for item in logical.samples)
        self.assertNotEqual(before, after)
        for delivery in permutations(logical.samples):
            received = update(make_learner(), replace(logical, samples=delivery))
            self.assertEqual(received.probabilities, result.probabilities)

    def test_capture_preserves_signed_zero_and_does_not_alias_tensors(self):
        for dtype, expected in ((torch.float32, NEGATIVE_ZERO), (torch.float64, 1 << 63)):
            with self.subTest(dtype=dtype):
                value = torch.tensor([-0.0], dtype=dtype)
                tokens = Tokens(**dict.fromkeys(ROLES, value), active=torch.tensor([True]))
                observed = capture("slash/name", tokens)
                value.fill_(1)
                self.assertEqual(observed.words, ((expected,),) * len(ROLES))
                self.assertEqual(observed.active, (True,))

    def test_actual_objective_check_requires_words_dtype_and_complete_response_mask(self):
        value = torch.tensor([-0.0, -0.0], dtype=torch.float32)
        tokens = Tokens(**dict.fromkeys(ROLES, value), active=torch.tensor([True, True]))
        observed = checked("a", tokens, advantage=-0.0, count=2)
        self.assertEqual(observed.words[ROLES.index("advantage")], (NEGATIVE_ZERO,) * 2)
        changes = [replace(tokens, advantage=torch.zeros_like(value)),
                   replace(tokens, advantage=value.double()),
                   replace(tokens, current=value[:1]),
                   replace(tokens, active=torch.tensor([True, False])),
                   replace(tokens, active=torch.tensor([1, 1]))]
        for changed in changes:
            with self.subTest(tokens=changed), self.assertRaises(ValueError):
                checked("a", changed, advantage=-0.0, count=2)
        for count in (0, 1, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                checked("a", tokens, advantage=-0.0, count=count)


if __name__ == "__main__":
    unittest.main()
