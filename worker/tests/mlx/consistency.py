import unittest

try:
    import mlx  # noqa: F401
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from pathlib import Path
import tempfile

import mlx.core as mx
import mlx.optimizers as optim

from worker.logical import Batch, Learner, Sample
from worker.mlx import adapter as mlx_adapter
from worker.mlx import backward as mlx_backward
from worker.mlx import learning as mlx_learning
from worker.mlx import numerics as mlx_numerics
from worker.mlx import inference as mlx_inference
from worker.mlx.crossscore import score
from worker.mlx import tensors as mlx_tensors
from worker.mlx.probability import words
from worker.mlx.rollout import generate, logprobs
from worker.record import ROLES
from worker.scalar import Profile
from worker.tests.mlx.crossscore import SAMPLING, loaded
from worker.batch import Reference
from worker.scoring import TokenPath
from worker.trajectory import Request

SCALE = 0.05
LONG = 12
PROMPT = " ".join(("one", "two", "three") * 24)


def shifted(parameters, seed):
    mx.random.seed(seed)
    return {name: value + SCALE * mx.random.normal(value.shape) if name.endswith("lora_b") else value
            for name, value in parameters.items()}


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = loaded(83)
        mlx_adapter.install(cls.runtime.model, shifted(mlx_adapter.state(cls.runtime.model), 89))
        cls.policy = mlx_adapter.state(cls.runtime.model)
        requests = tuple(Request(sample=str(index), group="long", prompt=PROMPT, seed=41 + index, limit=LONG, temperature=thermal)
                         for index, thermal in enumerate((0.8, 1.3, 0.6)))
        cls.trajectories = generate(cls.runtime.model, cls.runtime.tokenizer, requests, sampling=SAMPLING)

    def test_update_objective_consumes_the_sampled_words(self):
        proximal, fixed = mlx_learning.probabilities(self.trajectories, ((),) * len(self.trajectories))
        behavior = [words(item.behavior) for item in self.trajectories]
        self.assertEqual([words(value) for value in proximal], behavior)
        self.assertIs(fixed, proximal)
        samples = tuple(Sample(trajectory=item, proximal=old, reference=ref, advantage=advantage)
                        for item, old, ref, advantage in zip(self.trajectories, proximal, fixed, (1.0, -0.5, 0.25), strict=True))
        batch = Batch(samples=samples, order=tuple(item.request.sample for item in self.trajectories),
                      profile=Profile(epsilon=0.2, penalty=0.04))
        learner = Learner(model=self.runtime.model, optimizer=optim.AdamW(learning_rate=0.0001), evaluate=logprobs)
        try:
            with mlx_numerics.PRIMARY.learning(self.runtime.model):
                result = mlx_learning.update(learner, batch, linearize=mlx_backward.linearize)
        finally:
            mlx_adapter.install(self.runtime.model, self.policy)
        for observed, expected in zip(result.probabilities, behavior, strict=True):
            for role in ("behavior", "proximal", "reference", "current"):
                self.assertEqual(observed.words[ROLES.index(role)], expected)
        self.assertGreater(result.summary["gradient_norm"], 0)

    def test_rollout_scores_the_sampled_paths_under_the_reference(self):
        reference = shifted(self.policy, 97)
        directory = Path(tempfile.mkdtemp(prefix="invar-mlx-reference-"))
        digest = mlx_tensors.save_policy(directory / "reference.safetensors", reference)
        declared = Reference(adapter=directory / "reference.safetensors", digest=digest)
        _, scores = mlx_inference.scored(self.runtime, self.trajectories, reference=declared, sampling=SAMPLING)
        self.assertTrue(mlx_tensors.equal(mlx_adapter.state(self.runtime.model), self.policy))
        mlx_adapter.install(self.runtime.model, reference)
        try:
            paths = tuple(TokenPath(prefix=tuple(item.tokens[0, :item.prompt_length].tolist()),
                                    response=tuple(item.tokens[0, item.prompt_length:].tolist())) for item in self.trajectories)
            expected = score(self.runtime.model, self.runtime.tokenizer, tuple(item.request for item in self.trajectories),
                             paths=paths, sampling=SAMPLING)
        finally:
            mlx_adapter.install(self.runtime.model, self.policy)
        self.assertEqual(scores, tuple((digest, item.log_probability_bits) for item in expected))
        self.assertNotEqual([bits for _, bits in scores], [words(item.behavior) for item in self.trajectories])
