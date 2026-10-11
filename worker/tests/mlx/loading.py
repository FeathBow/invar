from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mlx.core as mx

from worker.implementation import INFERENCE
from worker.mlx import model as mlx_model
from worker.mlx import tensors
from worker.tests.mlx.crossscore import loaded
from worker.tests.mlx.rollout import model, tokenizer
from worker.tests.mlx.scoring import TEACHER_SEED

TEACHER = ("invar-bare-hybrid-test", "fixture-v1")


def checkpoint(path, *, lazy, strict):
    mx.random.seed(TEACHER_SEED)
    numerical, config = model(uniform_head=False, adapted=False)
    numerical.unfreeze()
    return numerical, config


class LoadingTests(unittest.TestCase):
    def test_an_empty_adapter_loads_the_declared_model_without_lora(self):
        root = Path(tempfile.mkdtemp(prefix="invar-mlx-bare-"))
        adapter = root / "adapter.safetensors"
        tensors.save_policy(adapter, {})
        expected = mlx_model.identities(loaded(TEACHER_SEED, adapted=False), INFERENCE)
        declared = root / "config.json"
        declared.write_text(json.dumps({"format": "invar-mlx-runtime-v1", "batch_size": 1, "prefill_step": 2,
                                        "cache_bytes": 1048576, "model": TEACHER[0], "revision": TEACHER[1]}))
        configuration = mlx_model.configuration(declared)
        records = []
        downloaded = mock.Mock(return_value=str(root))
        with mock.patch("huggingface_hub.snapshot_download", downloaded), mock.patch.object(mlx_model, "load_model", checkpoint), \
                mock.patch.object(mlx_model.operation, "load", return_value=tokenizer()), ExitStack() as scope:
            result = mlx_model.load(root, scope=scope, configuration=configuration, measure=lambda stage, action: action(),
                                    emit=lambda stage, values: records.append((stage, values)), initial=(adapter, expected))
        self.assertEqual((downloaded.call_args.args[0], downloaded.call_args.kwargs["revision"]), TEACHER)
        self.assertEqual(result.identity, TEACHER)
        self.assertEqual(mlx_model.identities(result, INFERENCE), expected)
        profile = dict(records)["profile"]
        self.assertEqual((profile["model"], profile["revision"], profile["native"]["adapter"]), (*TEACHER, None))


if __name__ == "__main__":
    unittest.main()
