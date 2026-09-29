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

from worker.logical import Learner, Plan
from worker.mlx import adapter as mlx_adapter
from worker.mlx import backward as mlx_backward
from worker.mlx import learning as mlx_learning
from worker.mlx import numerics as mlx_numerics
from worker.mlx import inference as mlx_inference
from worker.mlx.crossscore import score
from worker.mlx import tensors as mlx_tensors
from worker.mlx.probability import words
from worker.mlx.rollout import generate
from worker.mlx.training import logprobs
from worker.tests import scalar
from worker.tests.responder import Core
from worker.tests.mlx.crossscore import SAMPLING, loaded
from worker.batch import Reference
from worker.scoring import TokenPath
from worker.trajectory import Request
from worker.mlx import training as mlx_training

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

    def test_update_reports_learner_words_and_applies_the_core_cotangents(self):
        order = tuple(item.request.sample for item in self.trajectories)
        samples = {item.request.sample: (words(item.behavior), (), scalar.word(advantage))
                   for item, advantage in zip(self.trajectories, (1.0, -0.5, 0.25), strict=True)}
        core = Core(samples, (order,), scalar.Profile(epsilon=0.2, penalty=0.04))
        plan = Plan(trajectories={item.request.sample: item for item in self.trajectories}, steps=(order,), nonzero=3, exchange=core)
        learner = Learner(model=self.runtime.model, optimizer=optim.AdamW(learning_rate=0.0001), evaluate=logprobs)
        try:
            with mlx_training.learning(mlx_numerics.PRIMARY, self.runtime.model):
                self.runtime.model.train()
                linearized = {name: words(mlx_backward.linearize(self.runtime.model, item, evaluate=logprobs)[0])
                              for name, item in plan.trajectories.items()}
                result = mlx_learning.update(learner, plan, linearize=mlx_backward.linearize)
        finally:
            mlx_adapter.install(self.runtime.model, self.policy)
        self.assertEqual(result.proximal, linearized)
        self.assertEqual(result.currents, tuple((0, name, linearized[name]) for name in order))
        self.assertEqual(core.records[-1][4], tuple(core.answered))
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
