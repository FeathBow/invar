import hashlib
import io
import json
import math
from contextlib import redirect_stdout
from dataclasses import replace
from functools import partial
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from worker.tests.mlx.rollout import model, tokenizer

import mlx.core as mx

from worker import scalar
from worker.invocation import request as request_value
from worker.mlx import model as mlx_model
from worker.mlx import adapter as mlx_adapter
from worker.mlx import numerics, tensors
from worker.mlx import score as artifact
from worker.mlx.crossscore import PathSampler, TokenPath, score, validate
from worker.mlx.metrics import measure
from worker.mlx.probability import words
from worker.mlx.rollout import Sampler, Sampling, generate
from worker.trajectory import Request

SAMPLING = Sampling(batch_size=2, prefill_step=2)
ZERO_SUPPORT = 0xff800000


def loaded(seed):
    mx.random.seed(seed)
    numerical, config = model(uniform_head=False)
    numerics.PRIMARY.install(numerical)
    numerical.eval()
    return mlx_model.Loaded(model=numerical, tokenizer=tokenizer(), config=config,
                            identity=("invar-nonuniform-hybrid-test", str(seed)))


def requests():
    return tuple(Request(sample=str(index), group="paths", prompt=prompt,
                         seed=17 + index, limit=limit, temperature=thermal)
                 for index, (prompt, limit, thermal) in enumerate(
                     (("one", 4, 0.8), ("one two three", 6, 1.3), ("two", 5, 0.6))))


def path(trajectory):
    tokens = tuple(trajectory.tokens[0].tolist())
    return TokenPath(prefix=tokens[:trajectory.prompt_length], response=tokens[trajectory.prompt_length:])


def inspection(runtime, trajectory):
    value = {"binding": {"call": 0, "attempt": 0, "instance": 0},
             "tokens": trajectory.tokens[0].tolist(), "prompt_length": trajectory.prompt_length,
             "behavior_bits": list(words(trajectory.behavior)), "text": trajectory.text,
             "truncated": trajectory.truncated, "request": request_value(trajectory.request),
             "model": runtime.identity[0], "revision": runtime.identity[1],
             **mlx_model.identities(runtime)}
    value["log_sha256"] = hashlib.sha256(json.dumps(value).encode()).hexdigest()
    return json.dumps(value, allow_nan=False).encode()


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = loaded(71)
        cls.request = requests()[0]
        cls.trajectory, = generate(cls.runtime.model, cls.runtime.tokenizer, (cls.request,), sampling=SAMPLING)
        cls.encoded = inspection(cls.runtime, cls.trajectory)
        cls.expected = mlx_model.identities(cls.runtime)

    def test_actual_model_artifact_preserves_role_path_and_materialization(self):
        records = []
        measured = partial(measure, emit=lambda stage, values: records.append({"stage": stage, **values}))
        selected = artifact.source(self.encoded)
        result = artifact.observe(self.runtime, selected, expected=self.expected, sampling=SAMPLING, measured=measured)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual(result["source_inspection_sha256"], hashlib.sha256(self.encoded).hexdigest())
        self.assertEqual(result["source"], json.loads(self.encoded))
        self.assertEqual(result["log_probability_bits"], list(words(self.trajectory.behavior)))
        self.assertEqual(result["prefix_tokens"] + result["response_tokens"], self.trajectory.tokens[0].tolist())
        self.assertEqual({key: result["target"][key] for key in self.expected}, self.expected)
        self.assertEqual(result["role"], "cached_behavior_cross_score")
        self.assertEqual(result["use_admission"], "not_evaluated")
        self.assertNotIn("stage", result)
        self.assertEqual(result["execution"]["unused_native_lookahead_draws"], 1)
        self.assertEqual(result["probability"]["zero_support_word"], ZERO_SUPPORT)
        self.assertEqual(set(result["implementation"]["sources_sha256"]), set(artifact.IMPLEMENTATIONS))
        self.assertEqual([item["stage"] for item in records], ["verify_before", "cross_score", "verify_after"])
        self.assertTrue(all(item["peak_active"] > 0 and item["seconds"] >= 0 for item in records))

    def test_target_identity_and_termination_mismatches_fail(self):
        selected = artifact.source(self.encoded)
        measured = partial(measure, emit=lambda _stage, _values: None)
        for field in ("adapter", "tokenizer", "base", "assembly"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                artifact.observe(self.runtime, selected, expected={**self.expected, field: "0" * 64},
                                 sampling=SAMPLING, measured=measured)
        with self.assertRaisesRegex(ValueError, "stopping report"):
            artifact.observe(self.runtime, replace(selected, truncated=not selected.truncated),
                             expected=self.expected, sampling=SAMPLING, measured=measured)

    def test_offline_entry_loads_actual_checkpoint_and_emits_one_score(self):
        root = Path(tempfile.mkdtemp(prefix="invar-cached-score-"))
        source = root / "source.json"
        source.write_bytes(self.encoded)
        adapter = root / "adapter.safetensors"
        self.assertEqual(tensors.save_policy(adapter, mlx_adapter.state(self.runtime.model)), self.expected["adapter"])
        config = root / "config.json"
        config.write_text(json.dumps({"format": "invar-mlx-runtime-v1", "batch_size": 2,
                                     "prefill_step": 2, "cache_bytes": 1048576}))

        def tiny_loader(cache, *, scope, configuration, measure, emit, initial, numerics):
            self.assertEqual(cache, root)
            self.assertEqual(numerics, self.runtime.numerics)
            previous = mx.set_cache_limit(configuration.cache_bytes)
            scope.callback(mx.set_cache_limit, previous)
            return measure("load", lambda: mlx_model.activate(loaded(71), initial[0], expected=initial[1]))

        options = SimpleNamespace(path=source, cache=root, adapter=adapter, config=config,
                                  numerics="primary", probe_steps=None, digest=self.expected["adapter"],
                                  **{key + "_digest": self.expected[key] for key in ("tokenizer", "base", "assembly")})
        output = io.StringIO()
        with redirect_stdout(output):
            artifact.run(options, loader=tiny_loader)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["log_probability_bits"], list(words(self.trajectory.behavior)))
        self.assertEqual([value["stage"] for value in result["measurements"]],
                         ["load", "verify_before", "cross_score", "verify_after"])
        self.assertEqual(result["source_inspection_sha256"], hashlib.sha256(self.encoded).hexdigest())
        (root / "score.json").write_text(output.getvalue())
        print(json.dumps({"actual_tiny_checkpoint_score": str(root), "measurements": result["measurements"]}))

    def test_source_rejects_malformed_or_incomplete_observations(self):
        modifications = [
            {"unexpected": True}, {"behavior_bits": []}, {"prompt_length": True},
            {"tokens": [True, 2]}, {"truncated": 1}, {"adapter": "invalid"},
            {"binding": {"call": False, "attempt": 0, "instance": 0}},
            {"behavior_bits": [ZERO_SUPPORT] * len(words(self.trajectory.behavior))},
            {"request": {**request_value(self.request), "temperature": False}},
            {"request": {**request_value(self.request), "tokens": 0}},
        ]
        for update in modifications:
            with self.subTest(update=update), self.assertRaises(ValueError):
                artifact.source(json.dumps({**json.loads(self.encoded), **update}).encode())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            artifact.source(self.encoded[:-1] + b', "model": "duplicate"}')


class CachedPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = loaded(71)
        cls.candidate = loaded(97)
        cls.requests = requests()
        cls.generated = generate(cls.reference.model, cls.reference.tokenizer, cls.requests, sampling=SAMPLING)
        cls.paths = tuple(path(value) for value in cls.generated)

    def test_nonuniform_own_paths_reproduce_all_behavior_words(self):
        self.assertGreater(max(len(value.response) for value in self.paths), 2)
        self.assertGreater(len({word for value in self.generated for word in words(value.behavior)}), 1)
        before = {str(index): mx.array(value) for index, value in enumerate(mx.random.state)}
        identity = mlx_model.identities(self.reference)
        for sampling in (SAMPLING, Sampling(batch_size=1, prefill_step=3)):
            with self.subTest(sampling=sampling):
                observed = score(self.reference.model, self.reference.tokenizer, self.requests,
                                 paths=self.paths, sampling=sampling)
                self.assertEqual(tuple(value.request for value in observed), self.requests)
                self.assertEqual(tuple(value.path for value in observed), self.paths)
                for actual, original in zip(observed, self.generated, strict=True):
                    self.assertEqual(actual.log_probability_bits, words(original.behavior))
                    self.assertEqual(actual.truncated, original.truncated)
                    self.assertEqual(actual.lookahead_draws, 1)
        self.assertTrue(tensors.equal(before, {str(index): value for index, value in enumerate(mx.random.state)}))
        self.assertEqual(mlx_model.identities(self.reference), identity)

    def test_different_actual_model_scores_reference_paths_using_its_own_probabilities(self):
        freely = generate(self.candidate.model, self.candidate.tokenizer, self.requests, sampling=SAMPLING)
        self.assertNotEqual(tuple(path(value) for value in freely), self.paths)
        identity = mlx_model.identities(self.candidate)
        self.assertNotEqual(identity["base"], mlx_model.identities(self.reference)["base"])
        scored = score(self.candidate.model, self.candidate.tokenizer, self.requests,
                       paths=self.paths, sampling=SAMPLING)
        self.assertEqual(tuple(value.path for value in scored), self.paths)
        self.assertNotEqual(tuple(value.log_probability_bits for value in scored),
                            tuple(words(value.behavior) for value in self.generated))
        self.assertEqual(mlx_model.identities(self.candidate), identity)

    def test_paths_reject_wrong_prefix_vocabulary_termination_and_inventory(self):
        request = self.requests[0]
        valid = self.paths[0]
        invalid = [
            replace(valid, prefix=(0,)),
            replace(valid, response=(len(self.reference.tokenizer),) * request.limit),
            replace(valid, response=(self.reference.tokenizer.eos_token_id, 2)),
            replace(valid, response=(2,)),
            replace(valid, response=(2,) * (request.limit + 1)),
        ]
        for value in invalid:
            with self.subTest(path=value), self.assertRaises(ValueError):
                validate(request, value, self.reference.tokenizer)
        for supplied, paths in (((), ()), ((request,), ()), ((request,), (valid, valid))):
            with self.subTest(requests=supplied, paths=paths), self.assertRaises(ValueError):
                score(self.reference.model, self.reference.tokenizer, supplied, paths=paths, sampling=SAMPLING)
        for response in ([], (), (True,), (-1,), (1.0,)):
            with self.subTest(response=response), self.assertRaises(ValueError):
                TokenPath(prefix=valid.prefix, response=response)

    def test_forced_zero_support_is_preserved_and_lookahead_is_not_scored(self):
        request = replace(self.requests[0], limit=1, temperature=1.0)
        sampler = PathSampler(request, TokenPath(prefix=(2,), response=(0,)))
        distribution = mx.array([[-1000.0, 0.0]], dtype=mx.float32)
        self.assertEqual(sampler(distribution).item(), 0)
        observed = sampler.consume(0)
        self.assertEqual(words(observed.reshape(1)), (ZERO_SUPPORT,))
        self.assertEqual(observed.item(), -math.inf)
        with self.assertRaisesRegex(RuntimeError, "one lookahead"):
            sampler.completed()
        self.assertEqual(sampler(distribution).item(), 1)
        sampler.completed()
        with self.assertRaisesRegex(RuntimeError, "beyond"):
            sampler(distribution)
        freely = Sampler(request)
        self.assertEqual(freely(distribution).item(), 1)
        self.assertEqual(words(freely.consume(1).reshape(1)), (scalar.word(0.0),))

    def test_nonfinite_distribution_and_wrong_consumption_fail(self):
        sampler = PathSampler(self.requests[0], TokenPath(prefix=(2,), response=(0,)))
        sampler(mx.array([[math.nan, 0.0]], dtype=mx.float32))
        with self.assertRaisesRegex(RuntimeError, "actual sampling distribution"):
            sampler.consume(0)
        sampler(mx.array([[0.0, 0.0]], dtype=mx.float32))
        with self.assertRaisesRegex(RuntimeError, "actual sampling distribution"):
            sampler.consume(3)


if __name__ == "__main__":
    unittest.main()
