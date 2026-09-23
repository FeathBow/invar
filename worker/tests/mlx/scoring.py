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
        self.assertNotEqual(*values)

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
