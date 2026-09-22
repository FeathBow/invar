import unittest

from worker.vllm.context import response_history


class ContextTests(unittest.TestCase):
    def test_exact_empty_and_complete_histories_are_valid_in_both_modes(self):
        for asynchronous in (False, True):
            for expected in ((), (3,), (3, 5, 9)):
                with self.subTest(asynchronous=asynchronous, expected=expected):
                    self.assertIsNone(response_history(list(expected), expected, asynchronous=asynchronous))

    def test_async_native_placeholders_retain_length_and_every_known_word(self):
        expected = (3, 5, 9)
        for actual in ((-1, -1, -1), (3, -1, 9), (3, 5, -1)):
            with self.subTest(actual=actual):
                self.assertIsNone(response_history(actual, expected, asynchronous=True))

    def test_wrong_known_words_lengths_and_unexpected_placeholders_are_rejected(self):
        expected = (3, 5, 9)
        for asynchronous in (False, True):
            for actual in ((), (3,), (3, 5, 9, 1), (3, 6, 9), (3, -2, 9)):
                with self.subTest(asynchronous=asynchronous, actual=actual), self.assertRaisesRegex(ValueError, "context"):
                    response_history(actual, expected, asynchronous=asynchronous)
        with self.assertRaisesRegex(ValueError, "context"):
            response_history((3, -1, 9), expected, asynchronous=False)


if __name__ == "__main__":
    unittest.main()
