from contextlib import ExitStack
from types import SimpleNamespace
import unittest

import torch

from worker.hf.tensors import assert_equal
from worker.vllm.prescribed import Prescribed


def binding(limit):
    return SimpleNamespace(request=SimpleNamespace(limit=limit), eos=3)


def observation(*requests, active, history):
    return SimpleNamespace(pending=SimpleNamespace(rows=tuple(SimpleNamespace(request=key) for key in requests)),
                           sampling_requests=active, tokens=history)


class PrescribedTests(unittest.TestCase):
    def test_reordered_rows_and_incomplete_prefill_do_not_consume_the_wrong_path(self):
        control = Prescribed({"a": binding(2), "b": binding(2)}, {"a": [2, 0], "b": [1, 3]})
        history = {"a": [], "b": []}
        probabilities = torch.log_softmax(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]), dim=-1)
        before = probabilities.clone()
        tokens = torch.tensor([0, 1], dtype=torch.int64)
        selected = control.select(tokens, probabilities, monitor=observation("b", "a", active=("b",), history=history))
        self.assertEqual(selected.tolist(), [1, 1])
        self.assertEqual(tokens.tolist(), [0, 1])
        control.observe(selected.to(torch.int32).unsqueeze(-1), probabilities[:, 1:2])
        history["b"].append(1)
        selected = control.select(tokens, probabilities, monitor=observation("a", "b", active=("a", "b"), history=history))
        self.assertEqual(selected.tolist(), [2, 3])
        control.observe(selected.to(torch.int32).unsqueeze(-1), probabilities.gather(-1, selected.unsqueeze(-1)))
        history["a"].append(2)
        history["b"].append(3)
        with self.assertRaisesRegex(ValueError, "complete prescribed"):
            control.completed(history)
        selected = control.select(tokens[:1], probabilities[:1], monitor=observation("a", active=("a",), history=history))
        self.assertEqual(selected.tolist(), [0])
        control.observe(selected.to(torch.int32).unsqueeze(-1), probabilities[:1, :1])
        history["a"].append(0)
        control.completed(history)
        self.assertEqual(control.accounting("a"), {"native_sample_rows": 3, "ignored_prefill_rows": 1})
        self.assertEqual(control.accounting("b"), {"native_sample_rows": 2, "ignored_prefill_rows": 0})
        assert_equal(probabilities, before)

    def test_zero_support_and_signed_zero_require_exact_native_gather_words(self):
        for value in (-float("inf"), -0.0):
            with self.subTest(value=value):
                control = Prescribed({"a": binding(1)}, {"a": [2]})
                probabilities = torch.tensor([[-1., -2., value, -3.]], dtype=torch.float32)
                selected = control.select(torch.tensor([0]), probabilities,
                                          monitor=observation("a", active=("a",), history={"a": []}))
                self.assertEqual(selected.tolist(), [2])
                output = selected.to(torch.int32).unsqueeze(-1)
                with self.assertRaisesRegex(RuntimeError, "tensor mismatch"):
                    control.observe(output, torch.zeros((1, 1)))
                control.observe(output, probabilities[:, 2:3])
                with self.assertRaisesRegex(RuntimeError, "no prescribed"):
                    control.observe(output, probabilities[:, 2:3])

    def test_invalid_scope_boundaries_and_stale_history_are_rejected(self):
        for paths in ({}, {"a": []}, {"a": [True]}, {"a": [-1]}, {"a": [0]}, {"a": [3, 0]}, {"a": [0, 1, 2]}):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                Prescribed({"a": binding(2)}, paths)
        control = Prescribed({"a": binding(2)}, {"a": [2, 0]})
        probabilities = torch.log_softmax(torch.tensor([[1., 2., 3., 4.]]), dim=-1)
        for history in ({"a": [1]}, {"a": [2, 0]}):
            with self.subTest(history=history), self.assertRaisesRegex(ValueError, "cache history"):
                control.select(torch.tensor([0]), probabilities, monitor=observation("a", active=("a",), history=history))
        invalid = Prescribed({"a": binding(1)}, {"a": [4]})
        with self.assertRaisesRegex(ValueError, "output vocabulary"):
            invalid.select(torch.tensor([0]), probabilities, monitor=observation("a", active=("a",), history={"a": []}))

    def test_hook_restores_existing_instance_and_class_method_ownership(self):
        class SampleOwner(torch.nn.Module):
            sample = torch.log_softmax

        for instance_owned in (False, True):
            with self.subTest(instance_owned=instance_owned):
                sampler = SampleOwner()
                if instance_owned:
                    sampler.sample = torch.softmax
                original = sampler.sample
                control = Prescribed({"a": binding(1)}, {"a": [0]})
                monitor = SimpleNamespace(runner=SimpleNamespace(sampler=sampler))
                with self.assertRaisesRegex(ValueError, "denied"):
                    with ExitStack() as hooks:
                        control.attach(monitor, hooks)
                        self.assertIn("sample", vars(sampler))
                        self.assertIsNot(sampler.sample, original)
                        raise ValueError("denied")
                self.assertIs(sampler.sample, original)
                self.assertEqual("sample" in vars(sampler), instance_owned)


if __name__ == "__main__":
    unittest.main()
