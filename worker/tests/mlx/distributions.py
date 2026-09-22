import math
import unittest

import mlx.core as mx

from worker import scalar
from worker.distribution import Observed, Probe, Snapshot
from worker.mlx import model, tensors
from worker.mlx.crossscore import PathSampler, TokenPath, score
from worker.mlx.distribution import Capture
from worker.mlx.probability import tensor, words
from worker.mlx.rollout import Sampling, generate
from worker.tests.mlx.crossscore import SAMPLING, loaded, path, requests


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = loaded(71)
        cls.candidate = loaded(97)
        cls.requests = requests()
        cls.generated = generate(cls.reference.model, cls.reference.tokenizer, cls.requests, sampling=SAMPLING)
        cls.paths = tuple(path(value) for value in cls.generated)
        cls.probes = tuple(Probe(steps=tuple(sorted({0, len(value.response) - 1}))) for value in cls.paths)

    def test_actual_full_vectors_preserve_selected_words_rng_and_model_state(self):
        before = {str(index): mx.array(value) for index, value in enumerate(mx.random.state)}
        identity = model.identities(self.reference)
        collected = []
        for sampling in (SAMPLING, Sampling(batch_size=1, prefill_step=3)):
            measured = score(self.reference.model, self.reference.tokenizer, self.requests,
                             paths=self.paths, probes=self.probes, sampling=sampling)
            for actual, original, planned in zip(measured, self.generated, self.probes, strict=True):
                self.assertEqual(actual.log_probability_bits, words(original.behavior))
                self.assertEqual(actual.probe.probe, planned)
                self.assertEqual(tuple(value.step for value in actual.probe.snapshots), planned.steps)
                self.assertEqual(actual.probe.vocabulary, self.reference.config["vocab_size"])
                for snapshot in actual.probe.snapshots:
                    token = actual.path.response[snapshot.step]
                    selected = mx.log(tensor(snapshot.probability_bits)[token]).reshape(1)
                    self.assertEqual(words(selected), (actual.log_probability_bits[snapshot.step],))
                    self.assertGreater(len(set(snapshot.probability_bits)), 1)
                    self.assertAlmostEqual(math.fsum(map(scalar.number, snapshot.probability_bits)), 1, places=6)
            collected.append(tuple(value.probe for value in measured))
        self.assertEqual(*collected)
        self.assertTrue(tensors.equal(before, {str(index): value for index, value in enumerate(mx.random.state)}))
        self.assertEqual(model.identities(self.reference), identity)

    def test_distinct_model_captures_its_own_distributions_on_the_reference_prefixes(self):
        observations = []
        for runtime in (self.reference, self.candidate):
            identity = model.identities(runtime)
            actual = score(runtime.model, runtime.tokenizer, self.requests, paths=self.paths,
                           probes=self.probes, sampling=SAMPLING)
            self.assertEqual(tuple(value.path for value in actual), self.paths)
            observations.append(tuple(value.probe for value in actual))
            self.assertEqual(model.identities(runtime), identity)
        self.assertNotEqual(*observations)
        self.assertNotEqual(model.identities(self.reference)["base"], model.identities(self.candidate)["base"])

    def test_zero_support_is_retained_and_unused_lookahead_is_excluded(self):
        sampler = PathSampler(self.requests[0], TokenPath(prefix=(2,), response=(0,)), probe=Probe(steps=(0,)))
        logprobs = mx.array([[-1000.0, 0.0]], dtype=mx.float32)
        self.assertEqual(sampler(logprobs).item(), 0)
        self.assertEqual(words(sampler.consume(0).reshape(1)), (0xff800000,))
        self.assertEqual(sampler(logprobs).item(), 1)
        observed = sampler.completed()
        self.assertEqual(observed.snapshots, (Snapshot(step=0, probability_bits=(0, scalar.word(1))),))
        self.assertEqual(observed.vocabulary, 2)

    def test_probe_domain_rejects_invalid_or_out_of_path_steps(self):
        for steps in ((), [], (True,), (-1,), (0, 0), (2, 1), (0.5,)):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                Probe(steps=steps)
        for probes in ((), ("wrong",), (Probe(steps=(len(self.paths[0].response),)),)):
            with self.subTest(probes=probes), self.assertRaises(ValueError):
                score(self.reference.model, self.reference.tokenizer, self.requests[:1], paths=self.paths[:1],
                      probes=probes, sampling=SAMPLING)
        selected = Probe(steps=(0, 2))
        vector = Snapshot(step=0, probability_bits=(scalar.word(0.25), scalar.word(0.75)))
        with self.assertRaisesRegex(ValueError, "exactly"):
            Observed(probe=selected, snapshots=(vector,))

    def test_capture_rejects_nonfinite_incomplete_or_changing_distributions(self):
        for values in ((math.nan, 1), (-0.1, 1), (0, 0), (math.inf, 0), (1.1, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Capture(Probe(steps=(0,))).observe(0, mx.array([values], dtype=mx.float32))
        with self.assertRaisesRegex(ValueError, "FP32"):
            Capture(Probe(steps=(0,))).observe(0, mx.array([[0.5, 0.5]], dtype=mx.float16))
        for failure in ("duplicate", "vocabulary", "missing"):
            capture = Capture(Probe(steps=(0, 2)))
            capture.observe(0, mx.array([[0.5, 0.5]], dtype=mx.float32))
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                if failure == "duplicate":
                    capture.observe(0, mx.array([[0.5, 0.5]], dtype=mx.float32))
                elif failure == "vocabulary":
                    capture.observe(2, mx.array([[1.0]], dtype=mx.float32))
                else:
                    capture.completed()


if __name__ == "__main__":
    unittest.main()
