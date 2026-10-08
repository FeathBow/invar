import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from unittest import mock

import torch

from worker.hf import inference
from worker.hf.metrics import report
from worker.tests.hf.inference import fixture


class LoadingTests(unittest.TestCase):
    def test_the_production_loader_writes_the_model_records_to_the_transcript_it_is_given(self):
        runtime, identities = fixture()
        emitted = []

        def built(path, *, role, emit=report):
            emit("loading", {"missing_keys": []})
            emit("profile", {"model": "fixture"})
            return runtime.model

        def timed(stage, operation, *, emit=report):
            value = operation()
            emit(stage, {"seconds": 0.0})
            return value

        threads, precision, tf32 = torch.get_num_threads(), torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
        try:
            with torch.random.fork_rng(devices=[]), \
                    mock.patch("huggingface_hub.snapshot_download", return_value=str(runtime.adapter.parent)), \
                    mock.patch.object(inference, "load_model", built), mock.patch.object(inference, "measure", timed), \
                    mock.patch.object(inference, "load_tokenizer", return_value=runtime.tokenizer), \
                    mock.patch.object(inference, "activate"):
                inference.load(runtime.adapter.parent, runtime.adapter, expected=identities,
                               emit=lambda stage, values: emitted.append(stage))
        finally:
            torch.set_num_threads(threads)
            torch.set_float32_matmul_precision(precision)
            torch.backends.cudnn.allow_tf32 = tf32
        self.assertEqual(emitted, ["loading", "profile", "load"])


if __name__ == "__main__":
    unittest.main()
