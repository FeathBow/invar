from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest

import torch
from torch.overrides import _get_current_function_mode_stack

from worker.hf.tensors import assert_equal
from worker.vllm.distribution import Position, capture


def words(values):
    return tuple(int(value) & 0xffffffff for value in values.view(torch.int32).flatten().tolist())


def competition(logits, generator):
    reported = logits.log_softmax(dim=-1, dtype=torch.float32)
    mass = logits.softmax(dim=-1, dtype=torch.float32)
    noise = torch.empty_like(mass).exponential_(generator=generator)
    selected = mass.div_(noise).argmax(dim=-1)
    return selected, reported, mass


def threaded(barrier, logits):
    def operation():
        barrier.wait(timeout=10)
        return logits.softmax(dim=-1, dtype=torch.float32)
    result, observed = capture(operation, shape=tuple(logits.shape), positions=(Position(row=0, step=0),))
    return result, observed


class MassCaptureTests(unittest.TestCase):
    def test_selected_words_precede_inplace_competition_and_rng_is_preserved(self):
        logits = torch.tensor([[1., 2., 3., 4.], [-2., 1., 3., 0.]], dtype=torch.float32)
        expected = logits.softmax(dim=-1, dtype=torch.float32)
        before, after = (torch.Generator().manual_seed(719) for _ in range(2))
        baseline = competition(logits, before)
        positions = (Position(row=1, step=3), Position(row=0, step=0))
        actual, observed = capture(lambda: competition(logits, after), shape=(2, 4), positions=positions)
        assert_equal(actual, baseline)
        assert_equal(after.get_state(), before.get_state())
        self.assertFalse(torch.equal(actual[2], expected))
        for position, snapshot in observed:
            self.assertEqual(snapshot.step, position.step)
            self.assertEqual(snapshot.probability_bits, words(expected[position.row]))
        self.assertEqual(tuple(position for position, _ in observed), positions)

    def test_underflow_mass_is_observed_independently_of_finite_reported_log(self):
        logits = torch.tensor([[-120., 0.]], dtype=torch.float32)
        actual, observed = capture(lambda: competition(logits, torch.Generator().manual_seed(37)),
                                   shape=(1, 2), positions=(Position(row=0, step=4),))
        self.assertEqual(actual[1][0, 0].item(), -120.)
        _, snapshot = observed[0]
        self.assertEqual(snapshot.probability_bits, (0, 0x3f800000))

    def test_original_operator_output_is_returned_by_identity(self):
        logits = torch.tensor([[1., 2.]], dtype=torch.float32)
        outputs = []
        def operation():
            value = logits.softmax(-1, dtype=torch.float32)
            outputs.append(value)
            return value
        actual, _ = capture(operation, shape=(1, 2), positions=(Position(row=0, step=0),))
        self.assertIs(actual, outputs[0])

    def test_nested_observers_retain_the_same_actual_words(self):
        logits = torch.tensor([[1., 2., 3.]], dtype=torch.float32)
        positions = (Position(row=0, step=1),)
        def inner():
            return capture(lambda: logits.softmax(-1, dtype=torch.float32), shape=(1, 3), positions=positions)
        (actual, inner_words), outer_words = capture(inner, shape=(1, 3), positions=positions)
        self.assertEqual(inner_words, outer_words)
        self.assertEqual(inner_words[0][1].probability_bits, words(actual))

    def test_concurrent_threads_observe_only_their_own_scope(self):
        barrier = Barrier(2)
        inputs = (torch.tensor([[0., 2.]], dtype=torch.float32), torch.tensor([[3., 0.]], dtype=torch.float32))
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = tuple(executor.submit(threaded, barrier, logits) for logits in inputs)
            results = tuple(future.result(timeout=15) for future in pending)
        for logits, (actual, observed) in zip(inputs, results, strict=True):
            assert_equal(actual, logits.softmax(-1, dtype=torch.float32))
            self.assertEqual(observed[0][1].probability_bits, words(actual))
        self.assertNotEqual(results[0][1], results[1][1])

    def test_failed_operation_unwinds_and_propagates_the_original_exception(self):
        logits = torch.tensor([[0., 1.]], dtype=torch.float32)
        initial = _get_current_function_mode_stack()
        problem = ValueError("original sampler failure")
        def failing():
            logits.softmax(-1, dtype=torch.float32)
            raise problem
        with self.assertRaises(ValueError) as caught:
            capture(failing, shape=(1, 2), positions=(Position(row=0, step=0),))
        self.assertIs(caught.exception, problem)
        self.assertEqual(_get_current_function_mode_stack(), initial)
        actual, observed = capture(lambda: logits.softmax(-1, dtype=torch.float32),
                                   shape=(1, 2), positions=(Position(row=0, step=0),))
        self.assertEqual(observed[0][1].probability_bits, words(actual))

    def test_missing_and_ambiguous_capture_are_not_reconstructed(self):
        logits = torch.tensor([[0., 1.]], dtype=torch.float32)
        for operation in (lambda: logits.log_softmax(-1).exp(),
                          lambda: (logits.softmax(-1), logits.softmax(-1))):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                capture(operation, shape=(1, 2), positions=(Position(row=0, step=0),))

    def test_wrong_axis_dtype_and_declared_shape_are_rejected(self):
        logits = torch.tensor([[0., 1.], [2., 3.]], dtype=torch.float32)
        cases = ((lambda: logits.softmax(0), (2, 2)),
                 (lambda: logits.softmax(-1, dtype=torch.float64), (2, 2)),
                 (lambda: logits.softmax(-1), (1, 2)))
        for operation, shape in cases:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                capture(operation, shape=shape, positions=(Position(row=0, step=0),))

    def test_invalid_selection_is_rejected_before_operation(self):
        calls = []
        selected = Position(row=0, step=0)
        cases = (((1, 2), []), ((1, 2), ()), ((1, 2), (selected, selected)),
                 ((1, 2), (Position(row=1, step=0),)), ((True, 2), (selected,)),
                 ([1, 2], (selected,)))
        for shape, positions in cases:
            with self.subTest(shape=shape, positions=positions), self.assertRaises(ValueError):
                capture(lambda: calls.append(True), shape=shape, positions=positions)
        self.assertEqual(calls, [])
        for values in (dict(row=True, step=0), dict(row=0, step=-1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Position(**values)


if __name__ == "__main__":
    unittest.main()
