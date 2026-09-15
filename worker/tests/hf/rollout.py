import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

from worker.hf.rollout import reward


class RewardTests(unittest.TestCase):
    def test_numeric_answer_and_final_line(self):
        self.assertEqual(reward("Reasoning.\n#### +12.00", "#### 12", False), 1)
        self.assertEqual(reward("#### 12\nLater text", "#### 12", False), 0)
        self.assertEqual(reward("#### 11", "#### 12", False), 0)

    def test_truncation_and_malformed_responses(self):
        self.assertEqual(reward("#### 12", "#### 12", True), 0)
        for text in ("", "12", "#### 1e1", "#### NaN", "#### 1,000"):
            self.assertEqual(reward(text, "#### 12", False), 0)

    def test_invalid_ground_truth_is_not_a_zero_reward(self):
        with self.assertRaises(ValueError):
            reward("#### 12", "unparsed dataset answer", False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
