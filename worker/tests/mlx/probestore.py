from contextlib import ExitStack
import tempfile
import unittest

import mlx.core as mx

from worker.distribution import Probe
from worker.mlx import model, tensors
from worker.mlx.crossscore import PathSampler, score
from worker.mlx.rollout import Sampling, generate
from worker.probestore import Store
from worker.tests.mlx.crossscore import SAMPLING, loaded, path, requests
from worker.implementation import INFERENCE


class NativeProbeStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = loaded(71)
        cls.candidate = loaded(97)
        cls.requests = requests()
        cls.generated = generate(cls.reference.model, cls.reference.tokenizer, cls.requests, sampling=SAMPLING)
        cls.paths = tuple(path(value) for value in cls.generated)
        cls.probes = tuple(Probe(steps=tuple(sorted({0, len(value.response) - 1}))) for value in cls.paths)

    def check_capture(self, runtime, sampling):
        before = {str(index): mx.array(value) for index, value in enumerate(mx.random.state)}
        identity = model.identities(runtime, INFERENCE)
        memory = score(runtime.model, runtime.tokenizer, self.requests, paths=self.paths,
                       probes=self.probes, sampling=sampling)
        with ExitStack() as scope:
            stores = tuple(Store(scope.enter_context(tempfile.TemporaryFile()), probe=probe) for probe in self.probes)
            actual = score(runtime.model, runtime.tokenizer, self.requests, paths=self.paths,
                           probes=self.probes, stores=stores, sampling=sampling)
            for retained, expected in zip(actual, memory, strict=True):
                self.assertEqual(retained.path, expected.path)
                self.assertEqual(retained.request, expected.request)
                self.assertEqual(retained.log_probability_bits, expected.log_probability_bits)
                self.assertEqual(retained.truncated, expected.truncated)
                self.assertEqual(retained.lookahead_draws, expected.lookahead_draws)
                self.assertEqual(retained.probe.vocabulary, expected.probe.vocabulary)
                self.assertEqual(tuple(retained.probe.snapshots), expected.probe.snapshots)
        self.assertTrue(tensors.equal(before, {str(index): value for index, value in enumerate(mx.random.state)}))
        self.assertEqual(model.identities(runtime, INFERENCE), identity)

    def test_actual_reference_and_candidate_words_match_across_prefill_settings(self):
        for runtime in (self.reference, self.candidate):
            for sampling in (SAMPLING, Sampling(batch_size=1, prefill_step=3)):
                with self.subTest(base=model.identities(runtime, INFERENCE)["base"], sampling=sampling):
                    self.check_capture(runtime, sampling)

    def test_storage_cannot_change_the_declared_request_selection(self):
        with tempfile.TemporaryFile() as stream:
            wrong = Store(stream, probe=Probe(steps=(1,)))
            with self.assertRaisesRegex(ValueError, "selected probe"):
                PathSampler(self.requests[0], self.paths[0], probe=self.probes[0], store=wrong)
            with self.assertRaisesRegex(ValueError, "declared probe"):
                PathSampler(self.requests[0], self.paths[0], store=wrong)
        with self.assertRaisesRegex(ValueError, "store per request"):
            score(self.reference.model, self.reference.tokenizer, self.requests, paths=self.paths,
                  probes=self.probes, stores=(), sampling=SAMPLING)


if __name__ == "__main__":
    unittest.main()
