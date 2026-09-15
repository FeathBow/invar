import unittest

try:
    import mlx  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.optimizers as optim

from worker import core
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import codec as mlx_codec
from worker.mlx import tensors as mlx_tensors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StateTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-mlx-state-"))
        self.parameters = {"block.lora_a": mx.array([[1.0, -0.0]], dtype=mx.float32),
                           "block.lora_b": mx.array([[0.0], [1.0]], dtype=mx.float32)}
        self.policy = mlx_tensors.save_policy(self.directory / "adapter.safetensors", self.parameters)
        self.identities = {"adapter": self.policy, "tokenizer": "a" * 64,
                           "base": "b" * 64, "assembly": "c" * 64}
        self.optimizer = optim.AdamW(learning_rate=0.0001, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0, bias_correction=True)
        self.snapshot = mlx_checkpoint.observe(self.optimizer, identities=self.identities, parameters=self.parameters)

    def inspect(self, snapshot):
        checkpoint = self.directory / "learner.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        mlx_checkpoint.save(checkpoint, snapshot)
        fields = {"checkpoint": self.directory, "policy": self.policy, "learner": digest(checkpoint),
                  "reference-digest": self.policy, "rng-profile": "mlx", "initial-source": "provided",
                  "clip": "0.2", "penalty": "0", "delta": "0.0001", "rate": "0.0001",
                  "beta1": "0.9", "beta2": "0.999", "optimizer-epsilon": "0.00000001", "decay": "0",
                  **{key + "-digest": self.identities[key] for key in ("tokenizer", "base", "assembly")},
                  **{"behavior-" + key + "-digest": self.identities[key] for key in ("base", "assembly")}}
        session = mlx_codec.Session()
        result = core.exchange(["inspect", "initial", "--codec-mode", "stdio",
                                *[value for name, item in fields.items() for value in ("--" + name, item)]],
                               executable=os.environ.get("INVAR_CORE", "invar"), handler=session.handle)
        self.assertEqual((session.tensors, session.views), ([], {}))
        return result

    def test_actual_native_policy_and_rng_are_admitted(self):
        result = self.inspect(self.snapshot)
        self.assertEqual(result["policy"], self.policy)
        self.assertEqual(result["state"], {"steps": [], "mlx_rng_bytes": [8]})
        restored = mlx_tensors.policy(self.directory / "adapter.safetensors", self.policy)
        self.assertTrue(mlx_tensors.equal(self.parameters, restored))
        loaded = mlx_checkpoint.load((self.directory / "learner.pt").read_bytes())
        self.assertEqual(loaded["optimizer"]["state"]["step"].dtype, mx.uint64)
        self.assertEqual(loaded["rng"][0].dtype, mx.uint32)

    def test_default_uncorrected_adamw_is_rejected(self):
        self.optimizer.bias_correction = False
        observed = mlx_checkpoint.observe(self.optimizer, identities=self.identities, parameters=self.parameters)
        with self.assertRaisesRegex(ValueError, "MLX AdamW settings"):
            self.inspect(observed)

    def test_actual_learning_rate_tensor_is_checked(self):
        self.optimizer.learning_rate = 0.5
        observed = mlx_checkpoint.observe(self.optimizer, identities=self.identities, parameters=self.parameters)
        with self.assertRaisesRegex(ValueError, "Actual MLX learning rate"):
            self.inspect(observed)


if __name__ == "__main__":
    unittest.main()
