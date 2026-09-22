from decimal import Decimal, localcontext
from fractions import Fraction
import math
import unittest

from worker import probe, scalar
from worker.distribution import Snapshot

RELATIVE_CHECK_TOLERANCE = 2e-12


def decimal_kl(left, right):
    # An independent direct KL formula with exact FP32 inputs, rather than the
    # production generalized-KL summands and binary64 normalization.
    p = tuple(Fraction(scalar.number(word)) for word in left)
    q = tuple(Fraction(scalar.number(word)) for word in right)
    p_total, q_total = sum(p), sum(q)
    with localcontext() as context:
        context.prec = 100
        decimal = lambda value: Decimal(value.numerator) / Decimal(value.denominator)
        result = Decimal(0)
        for before, after in zip(p, q, strict=True):
            if before == 0:
                continue
            if after == 0:
                return Decimal("Infinity")
            a, b = before / p_total, after / q_total
            result += decimal(a) * decimal(a / b).ln()
        return result


class ReductionTests(unittest.TestCase):
    def test_finite_support_and_close_distributions_match_independent_decimal_kl(self):
        word = scalar.word
        cases = [
            ((word(0.5), word(0.5)), (word(0.5), word(0.5))),
            ((word(0.25), word(0.75)), (word(0.5), word(0.5))),
            ((word(0.5625), word(0.4375)), (word(0.5), word(0.5))),
            ((word(0.5 + 2**-24), word(0.5 - 2**-24)), (word(0.5), word(0.5))),
            ((word(1), 1), (word(1), 2)),
            ((word(1), 0), (word(1), 1)),
            ((0, word(1)), (word(1), 0)),
            ((word(0.1), word(0.2), word(0.3)), (word(0.2), word(0.3), word(0.4))),
        ]
        for left, right in cases:
            for before, after in ((left, right), (right, left)):
                with self.subTest(before=before, after=after):
                    p, _ = probe.normalize(Snapshot(step=0, probability_bits=before))
                    q, _ = probe.normalize(Snapshot(step=0, probability_bits=after))
                    measured = probe.kl(p, q)
                    expected = float(decimal_kl(before, after))
                    self.assertTrue(math.isclose(measured, expected, rel_tol=RELATIVE_CHECK_TOLERANCE, abs_tol=0), (measured, expected))
                    self.assertGreaterEqual(measured, 0)

    def test_zero_support_and_summary_preserve_infinity_as_data(self):
        rows = [{"step": 0, "kl_reference_candidate": probe.metric(0), "kl_candidate_reference": probe.metric(0.5)},
                {"step": 2, "kl_reference_candidate": probe.metric(math.inf), "kl_candidate_reference": probe.metric(1)}]
        summary = probe.summary(rows)
        self.assertEqual(summary["kl_reference_candidate"], {"mean": {"kind": "positive_infinity"},
                                                           "maximum": {"kind": "positive_infinity"}, "positive_infinity_steps": [2]})
        self.assertEqual(summary["kl_candidate_reference"]["mean"], {"kind": "finite", "value": 0.75})


if __name__ == "__main__":
    unittest.main()
