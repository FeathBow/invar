import io
import json
import os
from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mlx.core as mx

from worker.mlx import adapter, model, scoring, tensors
from worker.tests.mlx.crossscore import loaded

CORE = os.environ.get("INVAR_CORE", "invar")
SOURCE_ENTRY = Path(__file__).with_name("scoresource.py").resolve()
SCORE_ENTRY = Path(__file__).with_name("scorefixture.py").resolve()


def flags(values):
    return [str(item) for key, value in values.items() for item in ("--" + key, value)]


def load(cache, *, scope, configuration, measure, emit, initial=None):
    seed = json.loads((cache / "fixture.json").read_text())["seed"]
    previous = mx.set_cache_limit(configuration.cache_bytes)
    scope.callback(mx.set_cache_limit, previous)

    def prepare():
        runtime = loaded(seed)
        if initial is not None:
            model.activate(runtime, initial[0], expected=initial[1])
        return runtime

    return measure("load", prepare)


class BoundScoreFixture:
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="invar-bound-score-"))
        cls.config = cls.root / "config.json"
        cls.config.write_text(json.dumps({"format": "invar-mlx-runtime-v1", "batch_size": 1,
                                        "prefill_step": 2, "cache_bytes": 1048576}))
        cls.targets = []
        for seed in (71, 97):
            cache = cls.root / str(seed)
            cache.mkdir()
            (cache / "fixture.json").write_text(json.dumps({"seed": seed}))
            runtime = loaded(seed)
            identity = model.identities(runtime)
            checkpoint = cache / "adapter.safetensors"
            tensors.save_policy(checkpoint, adapter.state(runtime.model))
            cls.targets.append((cache, checkpoint, identity))
        cache, checkpoint, identity = cls.targets[0]
        cls.source_inputs = {"digest": identity["adapter"],
                             **{key + "-digest": identity[key] for key in ("tokenizer", "base", "assembly")},
                             "prompt": "one two three", "tokens": 6, "temperature": 0.8, "seed": 17,
                             "call": 7, "attempt": 7, "instance": 7}
        cls.source_log = cls.root / "source.jsonl"
        command = [CORE, "infer", *flags({**cls.source_inputs, "python": sys.executable,
                   "worker": SOURCE_ENTRY, "cache": cache, "adapter": checkpoint, "worker-config": cls.config})]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
        cls.source_log.write_text(completed.stdout)
        cls.source_log.with_suffix(".stderr").write_text(completed.stderr)
        (cls.root / "source-status.json").write_text(json.dumps({"exit_code": completed.returncode, "command": list(map(str, command))}))
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        cls.source_result = [json.loads(line) for line in completed.stdout.splitlines()][-1]

    def inputs(self, identity, *, call):
        return {**{"source-" + key: value for key, value in self.source_inputs.items()},
                "source-log": self.source_log, "source-exit-code": 0,
                "target-digest": identity["adapter"],
                **{"target-" + key + "-digest": identity[key] for key in ("tokenizer", "base", "assembly")},
                "call": call, "attempt": call, "instance": call}

    def command(self, label, arguments, *, success=True):
        completed = subprocess.run([str(CORE), *map(str, arguments)], capture_output=True, text=True, timeout=45)
        (self.root / (label + ".stdout")).write_text(completed.stdout)
        (self.root / (label + ".stderr")).write_text(completed.stderr)
        (self.root / (label + ".status.json")).write_text(json.dumps({"exit_code": completed.returncode, "arguments": list(map(str, arguments))}))
        if success:
            self.assertEqual(completed.returncode, 0, completed.stderr)
        else:
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
        return completed


class BoundScoringTests(BoundScoreFixture, unittest.TestCase):
    def test_actual_cached_scores_execute_and_reinspect_with_independent_path_sum(self):
        values = []
        for index, (cache, checkpoint, identity) in enumerate(self.targets):
            selected = self.inputs(identity, call=20 + index)
            result = self.command("score-" + str(index), ["score", *flags({**selected, "python": sys.executable,
                                  "worker": SCORE_ENTRY, "cache": cache, "adapter": checkpoint, "worker-config": self.config})])
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(events[-1]["stage"], "score_result")
            self.assertEqual([value["stage"] for value in events[-5:]],
                             ["consumed", "verify_before", "cross_score", "verify_after", "score_result"])
            observed = events[-1]["observation"]
            self.assertIn("worker.mlx.scoring", observed["implementation"]["sources_sha256"])
            self.assertEqual(observed["prefix_tokens"] + observed["response_tokens"], self.source_result["tokens"])
            path = self.root / ("score-" + str(index) + ".stdout")
            inspected = self.command("inspect-" + str(index), ["score", "inspect", *flags({**selected, "log": path, "exit-code": result.returncode})])
            admitted = json.loads(inspected.stdout)
            number = lambda word: Fraction(struct.unpack("=f", struct.pack("=I", word))[0])
            ratio = sum((number(p) - number(q) for p, q in zip(self.source_result["behavior_bits"], observed["log_probability_bits"], strict=True)), Fraction())
            self.assertEqual(admitted["source_minus_target_log_ratio"],
                             {"kind": "finite", "numerator": ratio.numerator, "denominator": ratio.denominator})
            self.assertEqual(admitted["use_admission"], "not_evaluated")
            if index == 0:
                self.assertEqual(observed["log_probability_bits"], self.source_result["behavior_bits"])
                self.assertEqual(ratio, 0)
            values.append(observed["log_probability_bits"])
            self.command("failed-exit-" + str(index), ["score", "inspect", *flags({**selected, "log": path, "exit-code": 7})], success=False)
            events[-1]["observation"]["probability"]["role"] = "rl_reference"
            wrong = self.root / ("wrong-role-" + str(index) + ".jsonl")
            wrong.write_text("\n".join(map(json.dumps, events)) + "\n")
            self.command("wrong-role-" + str(index), ["score", "inspect", *flags({**selected, "log": wrong, "exit-code": 0})], success=False)
        self.assertNotEqual(*values)
        self.compare_paths()
        print(json.dumps({"actual_bound_score_artifacts": str(self.root), "models": [71, 97]}))

    def compare_paths(self):
        cache, checkpoint, identity = self.targets[1]
        candidate_inputs = {**self.source_inputs, "digest": identity["adapter"],
                            **{key + "-digest": identity[key] for key in ("tokenizer", "base", "assembly")},
                            "call": 8, "attempt": 8, "instance": 8}
        candidate = self.command("candidate", ["infer", *flags({**candidate_inputs, "python": sys.executable,
                                 "worker": SOURCE_ENTRY, "cache": cache, "adapter": checkpoint, "worker-config": self.config})])
        candidate_log = self.root / "candidate.stdout"
        candidate_result = json.loads(candidate.stdout.splitlines()[-1])
        self.assertNotEqual(candidate_result["tokens"], self.source_result["tokens"])

        source_cache, source_checkpoint, source_identity = self.targets[0]
        reverse_inputs = {**{"source-" + key: value for key, value in candidate_inputs.items()},
                          "source-log": candidate_log, "source-exit-code": candidate.returncode,
                          "target-digest": source_identity["adapter"],
                          **{"target-" + key + "-digest": source_identity[key] for key in ("tokenizer", "base", "assembly")},
                          "call": 40, "attempt": 40, "instance": 40}
        backward = self.command("reverse-score", ["score", *flags({**reverse_inputs, "python": sys.executable,
                                "worker": SCORE_ENTRY, "cache": source_cache, "adapter": source_checkpoint,
                                "worker-config": self.config})])
        original = {**{"reference-" + key: value for key, value in self.source_inputs.items()},
                    "reference-log": self.source_log, "reference-exit-code": 0,
                    **{"candidate-" + key: value for key, value in candidate_inputs.items()},
                    "candidate-log": candidate_log, "candidate-exit-code": candidate.returncode}
        scores = {"reference-score-log": self.root / "score-1.stdout", "reference-score-exit-code": 0,
                  "reference-score-call": 21, "reference-score-attempt": 21, "reference-score-instance": 21,
                  "candidate-score-log": self.root / "reverse-score.stdout", "candidate-score-exit-code": backward.returncode,
                  "candidate-score-call": 40, "candidate-score-attempt": 40, "candidate-score-instance": 40}
        self.compare_findings(original, scores, (self.source_result, candidate_result))

    def compare_findings(self, original, scores, sources):
        number = lambda word: Fraction(struct.unpack("=f", struct.pack("=I", word))[0])
        for side, source in zip(("reference", "candidate"), sources, strict=True):
            path = scores[side + "-score-log"]
            body = json.loads(path.read_text().splitlines()[-1])["observation"]
            self.assertEqual(body["prefix_tokens"] + body["response_tokens"], source["tokens"])
            ratio = sum((number(p) - number(q) for p, q in zip(source["behavior_bits"], body["log_probability_bits"], strict=True)), Fraction())
            self.assertNotEqual(ratio, 0)
            budget = abs(ratio.numerator) // ratio.denominator + 1
            options = {**original, **scores, "relation": side + "-path-log-ratio", "budget": budget}
            result = self.command("compare-" + side, ["compare", "numerical", *flags(options)])
            actual = json.loads(result.stdout)
            self.assertIsNotNone(actual["observation"]["first_divergence_zero_based"])
            self.assertEqual(actual["finding"]["judgement"]["status"], "accept")
            self.assertEqual(actual["finding"]["use_admission"], "not_evaluated")
            self.assertEqual(len(actual["finding"]["judgement"]["assumptions"]), 14)
            measured = next(value for value in actual["observation"]["scored_paths"] if value["path_source"] == side.title())
            self.assertEqual(measured["source_minus_target_log_ratio"],
                             {"kind": "finite", "numerator": ratio.numerator, "denominator": ratio.denominator})
            rejected = self.command("compare-refute-" + side, ["compare", "numerical", *flags({**options, "budget": 0})])
            self.assertEqual(json.loads(rejected.stdout)["finding"]["judgement"]["status"], "refute")
            missing = {**original, "relation": side + "-path-log-ratio", "budget": budget}
            unknown = self.command("compare-missing-" + side, ["compare", "numerical", *flags(missing)])
            self.assertIn("MissingScoredPath", json.loads(unknown.stdout)["finding"]["judgement"]["reason"])
            self.command("compare-failed-score-" + side, ["compare", "numerical", *flags({**options, side + "-score-exit-code": 7})], success=False)
        kl = self.command("compare-kl", ["compare", "numerical", *flags({**original, **scores, "relation": "kl-reference-candidate", "probe-path": "reference", "budget": 1})])
        self.assertIn("MissingFullVocabulary", json.loads(kl.stdout)["finding"]["judgement"]["reason"])
        swapped = {**scores, "reference-score-log": scores["candidate-score-log"],
                   "reference-score-call": 40, "reference-score-attempt": 40, "reference-score-instance": 40}
        self.command("compare-wrong-path", ["compare", "numerical", *flags({**original, **swapped, "relation": "reference-path-log-ratio", "budget": 1})], success=False)
        orphan = {**original, "reference-score-call": 21, "relation": "tokens"}
        self.command("compare-orphan", ["compare", "numerical", *flags(orphan)], success=False)

    def test_missing_or_wrong_permission_prevents_score_execution(self):
        cache, checkpoint, identity = self.targets[0]
        planned = self.command("plan", ["score", "plan", *flags(self.inputs(identity, call=31))])
        call = json.loads(planned.stdout)
        for permission in ("", json.dumps({"binding": call["binding"], "program": "incorrect"}) + "\n"):
            incoming = io.StringIO(planned.stdout + permission)
            output = io.StringIO()
            options = SimpleNamespace(cache=cache, adapter=checkpoint, config=self.config)
            with self.assertRaises(ValueError):
                scoring.run(options, loader=load, source=incoming, output=output)
            stages = [json.loads(line)["stage"] for line in output.getvalue().splitlines()]
            self.assertEqual(stages[-1], "consumed")
            self.assertNotIn("cross_score", stages)
            self.assertNotIn("score_result", stages)


if __name__ == "__main__":
    unittest.main()
