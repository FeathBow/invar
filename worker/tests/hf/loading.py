import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from pathlib import Path
import tempfile
from unittest import mock

import torch
from safetensors.torch import save_file

from worker.hf import cohort, inference
from worker.hf.metrics import report
from worker.tests.hf.inference import IDENTITY, fixture, model
from worker.tests.hf.scoring import TEACHER, TEACHER_SEED, identities
from worker.tests.hf.tokenization import make_tokenizer


class LoadingTests(unittest.TestCase):
    def setUp(self):
        threads, precision, tf32 = torch.get_num_threads(), torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32

        def restore():
            torch.set_num_threads(threads)
            torch.set_float32_matmul_precision(precision)
            torch.backends.cudnn.allow_tf32 = tf32

        self.addCleanup(restore)

    def load(self, runtime, expected, **options):
        emitted, built = [], []

        def construct(path, *, role, emit=report, adapted, identity):
            built.append((adapted, identity))
            emit("loading", {"missing_keys": []})
            emit("profile", {"model": identity[0]})
            return runtime.model

        def timed(stage, operation, *, emit=report):
            value = operation()
            emit(stage, {"seconds": 0.0})
            return value

        downloaded = mock.Mock(return_value=str(runtime.adapter.parent))
        with torch.random.fork_rng(devices=[]), mock.patch("huggingface_hub.snapshot_download", downloaded), \
                mock.patch.object(inference, "load_model", construct), mock.patch.object(inference, "measure", timed), \
                mock.patch.object(inference, "load_tokenizer", return_value=runtime.tokenizer):
            loaded = inference.load(runtime.adapter.parent, runtime.adapter, expected=expected,
                                    emit=lambda stage, values: emitted.append(stage), **options)
        return loaded, emitted, built, downloaded

    def test_the_production_loader_writes_the_model_records_to_the_transcript_it_is_given(self):
        runtime, expected = fixture()
        with mock.patch.object(inference, "activate"):
            _, emitted, _, _ = self.load(runtime, expected, declared=IDENTITY)
        self.assertEqual(emitted, ["loading", "profile", "load"])

    def test_an_empty_adapter_loads_the_declared_model_without_lora(self):
        bare = model(seed=TEACHER_SEED, adapted=False)
        adapter = Path(tempfile.mkdtemp(prefix="invar-bare-")) / "adapter.safetensors"
        save_file({}, adapter)
        runtime = inference.Runtime(model=bare, tokenizer=make_tokenizer(), adapter=adapter, device="cpu", identity=IDENTITY)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        loaded, _, built, downloaded = self.load(runtime, identities(bare), declared=TEACHER)
        self.assertEqual(built, [(False, TEACHER)])
        self.assertEqual((downloaded.call_args.args[0], downloaded.call_args.kwargs["revision"]), TEACHER)
        self.assertEqual(loaded.identity, TEACHER)

    def test_a_resident_bare_model_activates_its_empty_adapter_again(self):
        bare = model(seed=TEACHER_SEED, adapted=False)
        adapter = Path(tempfile.mkdtemp(prefix="invar-bare-")) / "adapter.safetensors"
        save_file({}, adapter)
        expected = identities(bare)
        runtime = inference.Runtime(model=bare, tokenizer=make_tokenizer(), adapter=adapter, device="cpu", identity=TEACHER)
        activated = cohort.activate(cohort.activate(runtime, adapter, expected=expected), adapter, expected=expected)
        self.assertEqual(inference.materialized(activated, expected)["adapter"], expected["adapter"])

    def test_a_bare_model_with_changed_weights_is_refused_by_its_base_identity(self):
        bare = model(seed=TEACHER_SEED, adapted=False)
        expected = identities(bare)
        with torch.no_grad():
            next(bare.parameters()).view(-1)[0] += 1
        with self.assertRaisesRegex(RuntimeError, "frozen base binding mismatch"):
            inference.installed(bare, {}, expected)


if __name__ == "__main__":
    unittest.main()
