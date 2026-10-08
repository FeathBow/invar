import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import tempfile
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from worker import float32, session
from worker.hf import inference, score
from worker.hf.model import adapter_state
from worker.hf.policy import activate
from worker.hf.rollout import generate
from worker.hf.tensors import digest
from worker.implementation import INFERENCE
from worker.session import Reference
from worker.tests.hf.inference import TEST_THREADS, envelope, fixture
from worker.trajectory import Request

REQUEST = Request(sample="inference", group="inference", prompt="Compute the answer.", seed=1326, limit=4, temperature=0.8)


def words(values):
    return [float32.word(value) for value in values.tolist()]


def shifted(runtime):
    generator = torch.Generator().manual_seed(3)
    state = {name: value + 0.5 * torch.randn(value.shape, generator=generator) if "lora_B" in name else value.clone()
             for name, value in adapter_state(runtime.model).items()}
    path = Path(tempfile.mkdtemp(prefix="invar-reference-")) / "adapter.safetensors"
    save_file(state, path)
    return Reference(adapter=path, digest=digest(state)), state


class ReferenceScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(TEST_THREADS)

    def test_the_forced_path_under_the_sampling_policy_repeats_its_behavior_words(self):
        runtime, _ = fixture()
        trajectory = generate(runtime.model, runtime.tokenizer, REQUEST, device="cpu")
        self.assertEqual(words(score.score(runtime.model, trajectory, device="cpu")), words(trajectory.behavior))

    def test_reference_words_follow_the_recorded_path_under_the_reference_and_the_policy_is_restored(self):
        runtime, identities = fixture()
        trajectory = generate(runtime.model, runtime.tokenizer, REQUEST, device="cpu")
        reference, state = shifted(runtime)
        (scored,) = score.referenced(runtime, (trajectory,), reference, identities=identities)
        self.assertEqual(digest(adapter_state(runtime.model)), identities["adapter"])
        other, _ = fixture()
        activate(other.model, state, base=identities["base"], assembly=identities["assembly"], role=INFERENCE)
        self.assertEqual(scored, (reference.digest, tuple(words(score.score(other.model, trajectory, device="cpu")))))
        self.assertNotEqual(list(scored[1]), words(trajectory.behavior))

    def test_a_failed_reference_scoring_restores_the_policy_and_a_failed_restore_is_not_hidden(self):
        runtime, identities = fixture()
        trajectory = generate(runtime.model, runtime.tokenizer, REQUEST, device="cpu")
        reference, _ = shifted(runtime)
        with mock.patch.object(score, "score", side_effect=RuntimeError("scoring failed")):
            with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                score.referenced(runtime, (trajectory,), reference, identities=identities)
        self.assertEqual(digest(adapter_state(runtime.model)), identities["adapter"])
        with self.assertRaisesRegex(ValueError, "do not match"):
            score.referenced(runtime, (trajectory,), Reference(adapter=reference.adapter, digest="0" * 64), identities=identities)
        self.assertEqual(digest(adapter_state(runtime.model)), identities["adapter"])
        with mock.patch.object(score, "activate", side_effect=[None, RuntimeError("restore failed")]):
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                score.referenced(runtime, (trajectory,), reference, identities=identities)

    def test_a_call_whose_reference_scoring_fails_reports_no_result(self):
        runtime, identities = fixture()
        reference, _ = shifted(runtime)
        stages = []
        with mock.patch.object(score, "score", side_effect=RuntimeError("scoring failed")):
            with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                inference.execute(runtime, session.decode(envelope(identities, 0)), approve=lambda invocation: None,
                                  measure=lambda stage, action: action(), emit=lambda stage, values: stages.append(stage),
                                  reference=reference)
        self.assertNotIn("result", stages)
        self.assertEqual(digest(adapter_state(runtime.model)), identities["adapter"])


if __name__ == "__main__":
    unittest.main()
