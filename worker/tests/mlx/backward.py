import unittest

try:
    import mlx  # noqa: F401
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

import mlx.core as mx
import mlx.nn as nn

from worker.mlx import adapter as mlx_adapter
from worker.mlx import backward as mlx_backward
from worker.mlx import learning as mlx_learning
from worker.mlx import probability as mlx_probability
from worker.mlx import numerics as mlx_numerics
from worker.mlx import rollout as mlx_rollout
from worker.mlx import tensors as mlx_tensors
from worker.tests.mlx.rollout import model, WORDS
from worker.trajectory import Request, Trajectory
from worker.mlx import training as mlx_training

TOKEN_COUNT = 37
PROMPT_COUNT = 4
PARAMETER_SCALE = 0.01


class BackwardTests(unittest.TestCase):
    def test_layerwise_vjp_preserves_all_hybrid_parameter_gradients(self):
        mx.random.seed(29)
        numerical, config = model()
        projection = nn.Linear(config["hidden_size"], len(WORDS), bias=False)
        projection.set_dtype(mx.bfloat16)
        numerical.language_model.lm_head = nn.QuantizedLinear.from_linear(projection, group_size=64, bits=4)
        numerical.language_model.lm_head.freeze()
        parameters = {name: mx.random.normal(value.shape) * PARAMETER_SCALE if name.endswith("lora_b") else value
                      for name, value in mlx_adapter.state(numerical).items()}
        mlx_adapter.install(numerical, parameters)
        mlx_numerics.PRIMARY.install(numerical)
        numerical.train()
        request = Request(sample="native-reverse", group="hybrid", prompt="one", seed=31,
                          limit=TOKEN_COUNT - PROMPT_COUNT, temperature=0.8)
        trajectory = Trajectory(request=request, tokens=(mx.arange(TOKEN_COUNT) % len(WORDS))[None, :],
                                prompt_length=PROMPT_COUNT, behavior=mx.zeros((request.limit,)),
                                text="", truncated=True)
        cotangent = mx.linspace(-0.2, 0.3, request.limit)
        with mlx_training.learning(mlx_numerics.PRIMARY, numerical):
            self.compare(numerical, trajectory, cotangent=cotangent, parameters=parameters)
        self.assertTrue(mlx_tensors.equal(mlx_adapter.state(numerical), parameters))

    def compare(self, numerical, trajectory, *, cotangent, parameters):
        reference, whole = mlx_learning.linearize(numerical, trajectory, evaluate=mlx_training.logprobs)
        current, layered = mlx_backward.linearize(numerical, trajectory, evaluate=mlx_training.logprobs)
        self.assertEqual(mlx_probability.words(current), mlx_probability.words(reference))
        expected = whole(cotangent=cotangent)
        actual = layered(cotangent=cotangent)
        self.assertGreater(mlx_learning.norm(expected), 0)
        self.assertEqual(actual.keys(), parameters.keys())
        for name in parameters:
            with self.subTest(parameter=name):
                self.assertTrue(mlx_tensors.equal({name: actual[name]}, {name: expected[name]}))


if __name__ == "__main__":
    unittest.main()
