import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest
from dataclasses import replace

from worker.advantage import Reward, advantages, check, word
from worker.cohort import decode
from worker.tests.hf.cohort import request

NEGATIVE_ZERO = 0x80000000
POSITIVE_UNIT_GROUP = 0x3F7FF2E5
NEGATIVE_UNIT_GROUP = 0xBF7FF2E5
DECIMAL_RESIDUAL = 0xAA1C4000
DELTA = 1e-4


def rewards(values):
    return tuple(Reward(sample=str(index), group="g", value=value) for index, value in enumerate(values))


class AdvantageTests(unittest.TestCase):
    def test_fixed_words_include_signed_zero_and_decimal_residual(self):
        cases = [((0.0, 1.0), (NEGATIVE_UNIT_GROUP, POSITIVE_UNIT_GROUP)),
                 ((0.0, 0.0), (0, 0)), ((1.0, 1.0), (0, 0)),
                 ((-0.0, 0.0), (NEGATIVE_ZERO, 0)),
                 ((0.1, 0.1, 0.1), (DECIMAL_RESIDUAL,) * 3)]
        for values, expected in cases:
            with self.subTest(values=values):
                actual = advantages(rewards(values), DELTA)
                self.assertEqual(tuple(word(value) for _, value in actual), expected)

    def test_precheck_recomputes_values_and_preserves_logical_association(self):
        original = decode(request())
        checked = check(original)
        self.assertEqual(checked.values, (("a", -1 / (1 + 2 * DELTA)),
                                          ("b", 1 / (1 + 2 * DELTA))))
        self.assertEqual(check(replace(original, samples=original.samples[::-1])).values, checked.values)
        changed = replace(original.samples[0], advantage_bits=NEGATIVE_UNIT_GROUP)
        with self.assertRaisesRegex(ValueError, "core expectation"):
            check(replace(original, samples=(changed, original.samples[1])))

    def test_regrouping_known_rewards_is_an_effective_mutation(self):
        original = tuple(Reward(sample=label, group=group, value=value)
                         for label, group, value in (("a", "first", 0), ("b", "first", 0),
                                                     ("c", "second", 1), ("d", "second", 1)))
        grouped = advantages(original, DELTA)
        changed = tuple(replace(item, group=str(index % 2)) for index, item in enumerate(original))
        self.assertEqual(tuple(word(value) for _, value in grouped), (0,) * 4)
        self.assertEqual(tuple(word(value) for _, value in advantages(changed, DELTA)),
                         (NEGATIVE_UNIT_GROUP,) * 2 + (POSITIVE_UNIT_GROUP,) * 2)

    def test_numerical_overflow_is_exposed(self):
        for values in ((1e308, 1e308, -1e308), (-1e200, 1e200),
                       (float("nan"), 0), (float("inf"), 0)):
            with self.subTest(values=values), self.assertRaises((ValueError, OverflowError)):
                advantages(rewards(values), DELTA)
        for value in (float("nan"), float("inf"), 1e100):
            with self.subTest(value=value), self.assertRaises((ValueError, OverflowError)):
                word(value)


if __name__ == "__main__":
    unittest.main()
