import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch

from worker.hf import assembly
from worker.hf import frozen
from worker.tests.hf.checkpoint import square_learner
from worker.hf.learning import update
from worker.tests.hf.learning import batch, make_learner
from worker.hf.policy import activate
from worker.hf.model import adapter_state
from worker.hf.tensors import assert_equal
from worker.hf.checkpoint import restore
from worker.hf.step import checkpoint_update
from worker.tests.hf.step import prepared
from worker.tests.hf.successor import observed
from worker.tests.hf.tokenization import make_tokenizer
from worker.implementation import LEARNING

FRACTIONAL_INPUT = ((1 / 3, 1 / 7),)
ADAPTER_SHIFT = 1
TEST_THREADS = 2


@contextmanager
def flag(owner, name):
    previous = getattr(owner, name)
    setattr(owner, name, not previous)
    try:
        yield
    finally:
        setattr(owner, name, previous)


@contextmanager
def precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if previous == "highest" else "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@contextmanager
def determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(not enabled, warn_only=warning)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warning)


@contextmanager
def warning_mode():
    enabled = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled, warn_only=not warning)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warning)


@contextmanager
def default_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64 if previous == torch.float32 else torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def configurations():
    return (("matmul precision", precision()),
            ("default dtype", default_dtype()),
            ("FP16 reduction", flag(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction")),
            ("BF16 reduction", flag(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction")),
            ("FP16 accumulation", flag(torch.backends.cuda.matmul, "allow_fp16_accumulation")),
            ("cuDNN enabled", flag(torch.backends.cudnn, "enabled")),
            ("cuDNN TF32", flag(torch.backends.cudnn, "allow_tf32")),
            ("cuDNN benchmark", flag(torch.backends.cudnn, "benchmark")),
            ("cuDNN deterministic", flag(torch.backends.cudnn, "deterministic")),
            ("deterministic algorithms", determinism()),
            ("deterministic warning", warning_mode()),
            ("CPU autocast", torch.autocast("cpu", dtype=torch.bfloat16)))


class BackendTests(unittest.TestCase):
    def test_actual_autocast_changes_forward_without_changing_model_contents(self):
        model, _ = square_learner()
        inputs = torch.tensor(FRACTIONAL_INPUT, dtype=torch.float32)
        identity, base, state = assembly.digest(model, LEARNING), frozen.digest(model), adapter_state(model)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(inputs)
            self.assertEqual(base, frozen.digest(model))
            assert_equal(state, adapter_state(model))
            self.assertNotEqual(identity, assembly.digest(model, LEARNING))
        self.assertEqual(identity, assembly.digest(model, LEARNING))

    def test_backend_mismatch_prevents_actual_adapter_activation(self):
        learner = make_learner()
        tokenizer = make_tokenizer()
        expected_base, expected_assembly = frozen.digest(learner.model), assembly.digest(learner.model, LEARNING)
        state = {name: value + ADAPTER_SHIFT for name, value in adapter_state(learner.model).items()}
        for name, context in configurations():
            with self.subTest(setting=name), context:
                before = observed(learner, tokenizer)
                with self.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
                    activate(learner.model, state, base=expected_base, assembly=expected_assembly, role=LEARNING)
                assert_equal(before, observed(learner, tokenizer))

    def test_backend_mismatch_prevents_checkpoint_restoration(self):
        learner, _, _, options = prepared()
        update(learner, batch())
        for name, context in configurations():
            with self.subTest(setting=name), context:
                before = observed(learner, options.tokenizer)
                with self.assertRaisesRegex(RuntimeError, "model assembly binding mismatch"):
                    restore(learner.model, learner.optimizer, options.checkpoint, tokenizer=options.tokenizer)
                assert_equal(before, observed(learner, options.tokenizer))

    def test_backend_mismatch_cannot_write_a_bound_successor(self):
        learner, _, request, options = prepared()
        result = update(learner, batch())
        for name, context in configurations():
            with self.subTest(setting=name), context:
                before = observed(learner, options.tokenizer)
                output = Path(tempfile.mkdtemp(prefix="invar-backend-successor-"))
                with self.assertRaisesRegex(RuntimeError, "Checkpoint identity differs"):
                    checkpoint_update(learner, request, output, tokenizer=options.tokenizer, summary=result.summary)
                self.assertEqual(list(output.iterdir()), [])
                assert_equal(before, observed(learner, options.tokenizer))


if __name__ == "__main__":
    torch.set_num_threads(TEST_THREADS)
    unittest.main()
