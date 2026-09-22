import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import io
import json
import math
import sys
from types import SimpleNamespace
import unittest

from worker import probe
from worker.mlx import scoring
from worker.tests.mlx.scoring import BoundScoreFixture, SCORE_ENTRY, SOURCE_ENTRY, flags, load
from worker.tests.probe import RELATIVE_CHECK_TOLERANCE, decimal_kl

FIXTURE_KL_BUDGET = 100


def rational(value):
    return Fraction(value["numerator"], value["denominator"])


class BoundProbeTests(BoundScoreFixture, unittest.TestCase):
    def test_actual_full_vectors_execute_and_reinspect_on_both_targets(self):
        steps = [0, len(self.source_result["behavior_bits"]) - 1]
        self.assertLess(*steps)
        artifacts = []
        for index, (cache, checkpoint, identity) in enumerate(self.targets):
            selected = {**self.inputs(identity, call=50 + index), "probe-steps": json.dumps(steps)}
            result = self.command("probe-" + str(index), ["score", *flags({**selected, "python": sys.executable,
                                  "worker": SCORE_ENTRY, "cache": cache, "adapter": checkpoint, "worker-config": self.config})])
            events = [json.loads(line) for line in result.stdout.splitlines()]
            consumed = next(value for value in events if value["stage"] == "consumed")
            self.assertEqual(consumed["probe_steps"], steps)
            observed = events[-1]["observation"]
            self.assertEqual(observed["source_inspection"], consumed["source_inspection"])
            self.assertEqual(observed["full_vocabulary"]["steps"], steps)
            self.assertEqual(observed["measurements"], [value for value in events if value["stage"] in
                             ("loading", "profile", "load", "verify_before", "cross_score", "verify_after")])
            self.assertTrue({"worker.mlx.scoring", "worker.mlx.distribution", "worker.distribution"}.issubset(
                            observed["implementation"]["sources_sha256"]))
            path = self.root / ("probe-" + str(index) + ".stdout")
            inspected = self.command("inspect-probe-" + str(index), ["score", "inspect", *flags({**selected, "log": path, "exit-code": 0})])
            admitted = json.loads(inspected.stdout)
            self.assertEqual(admitted["observation"], observed)
            self.assertEqual(admitted["strength"], "finite_full_vocabulary_observation")
            self.assertEqual(admitted["use_admission"], "not_evaluated")
            encoded = json.dumps(observed, allow_nan=False).encode()
            (self.root / (str(index) + "-probe.json")).write_bytes(encoded)
            artifacts.append(probe.read(encoded))
            if index == 0:
                self.assertEqual(observed["log_probability_bits"], self.source_result["behavior_bits"])
            self.command("wrong-probe-steps-" + str(index), ["score", "inspect", *flags({**selected,
                         "probe-steps": "[0]", "log": path, "exit-code": 0})], success=False)
            plain = {key: value for key, value in selected.items() if key != "probe-steps"}
            self.command("missing-probe-plan-" + str(index), ["score", "inspect", *flags({**plain,
                         "log": path, "exit-code": 0})], success=False)
            self.reject_changed_snapshot(selected, events, index=index)
        compared = probe.compare(*artifacts)
        for row, left, right in zip(compared["observations"], artifacts[0].vectors.snapshots,
                                   artifacts[1].vectors.snapshots, strict=True):
            for name, p, q in (("kl_reference_candidate", left, right), ("kl_candidate_reference", right, left)):
                expected = decimal_kl(p.probability_bits, q.probability_bits)
                self.assertTrue(math.isclose(row[name]["value"], float(expected),
                                rel_tol=RELATIVE_CHECK_TOLERANCE, abs_tol=0), (name, row, expected))
        (self.root / "comparison.json").write_text(json.dumps(compared, allow_nan=False))
        self.check_findings(artifacts, steps=steps)
        print(json.dumps({"actual_bound_probe_artifacts": str(self.root), "models": [71, 97], "steps": steps}))

    def check_findings(self, artifacts, *, steps):
        cache, checkpoint, identity = self.targets[1]
        candidate_inputs = {**self.source_inputs, "digest": identity["adapter"],
                            **{key + "-digest": identity[key] for key in ("tokenizer", "base", "assembly")},
                            "call": 8, "attempt": 8, "instance": 8}
        candidate = self.command("candidate", ["infer", *flags({**candidate_inputs, "python": sys.executable,
                                 "worker": SOURCE_ENTRY, "cache": cache, "adapter": checkpoint, "worker-config": self.config})])
        original = {**{"reference-" + key: value for key, value in self.source_inputs.items()},
                    **{"candidate-" + key: value for key, value in candidate_inputs.items()},
                    "reference-log": self.source_log, "reference-exit-code": 0,
                    "candidate-log": self.root / "candidate.stdout", "candidate-exit-code": candidate.returncode}
        attached = {"probe-path": "reference", "probe-steps": json.dumps(steps)}
        for side, call in (("reference", 50), ("candidate", 51)):
            attached.update({side + "-probe-log": self.root / ("probe-" + str(call-50) + ".stdout"),
                             side + "-probe-exit-code": 0,
                             **{side + "-probe-" + key: call for key in ("call", "attempt", "instance")}})
        for direction in ("reference-candidate", "candidate-reference"):
            options = {**original, **attached, "relation": "kl-" + direction, "budget": FIXTURE_KL_BUDGET}
            result = self.command("kl-" + direction, ["compare", "numerical", *flags(options)])
            actual = json.loads(result.stdout)
            self.assertIsNotNone(actual["observation"]["first_divergence_zero_based"])
            self.assertEqual(actual["finding"]["judgement"]["status"], "accept")
            self.assertEqual(len(actual["finding"]["judgement"]["assumptions"]), 16)
            self.assertEqual(actual["finding"]["use_admission"], "not_evaluated")
            self.check_enclosures(actual["observation"]["full_vocabulary"][0], artifacts)
            rejected = self.command("kl-zero-" + direction, ["compare", "numerical", *flags({**options, "budget": 0})])
            self.assertEqual(json.loads(rejected.stdout)["finding"]["judgement"]["status"], "refute")
            missing = {key: value for key, value in options.items() if not key.startswith("candidate-probe-")}
            unknown = self.command("kl-missing-" + direction, ["compare", "numerical", *flags(missing)])
            self.assertIn("MissingFullVocabulary", json.loads(unknown.stdout)["finding"]["judgement"]["reason"])
            self.check_boundary(options, actual, direction=direction)
        self.command("kl-wrong-path", ["compare", "numerical", *flags({**options, "probe-path": "candidate"})], success=False)
        self.command("kl-failed-probe", ["compare", "numerical", *flags({**options, "reference-probe-exit-code": 7})], success=False)
        self.command("kl-wrong-target", ["compare", "numerical", *flags({**options,
                     "reference-probe-log": attached["candidate-probe-log"],
                     "reference-probe-call": 51, "reference-probe-attempt": 51, "reference-probe-instance": 51})], success=False)
        unspecified = {key: value for key, value in options.items() if key != "probe-path"}
        self.command("kl-unspecified-path", ["compare", "numerical", *flags(unspecified)], success=False)

    def check_enclosures(self, observed, artifacts):
        self.assertEqual(observed["path_source"], "Reference")
        self.assertEqual(observed["aggregation"], "every selected step must meet the budget")
        for row, left, right in zip(observed["steps"], artifacts[0].vectors.snapshots, artifacts[1].vectors.snapshots, strict=True):
            self.assertEqual(row["step"], left.step)
            for direction, p, q in (("reference_candidate", left, right), ("candidate_reference", right, left)):
                bounds = row["kl_" + direction]
                expected = Fraction(decimal_kl(p.probability_bits, q.probability_bits))
                self.assertLessEqual(rational(bounds["lower"]), expected)
                self.assertLessEqual(expected, rational(bounds["upper"]))

    def check_boundary(self, options, actual, *, direction):
        rows = actual["observation"]["full_vocabulary"][0]["steps"]
        key = "kl_" + direction.replace("-", "_")
        lower = max(rational(row[key]["lower"]) for row in rows)
        upper = max(rational(row[key]["upper"]) for row in rows)
        self.assertLess(lower, upper)
        with localcontext() as context:
            context.prec = 100
            midpoint = (lower + upper) / 2
            budget = str(Decimal(midpoint.numerator) / Decimal(midpoint.denominator))
        self.assertLess(lower, Fraction(Decimal(budget)))
        self.assertLess(Fraction(Decimal(budget)), upper)
        unresolved = self.command("kl-boundary-" + direction, ["compare", "numerical", *flags({**options, "budget": budget})])
        self.assertIn("KLReductionUncertain", json.loads(unresolved.stdout)["finding"]["judgement"]["reason"])

    def reject_changed_snapshot(self, selected, events, *, index):
        mutations = (lambda body: body["full_vocabulary"]["snapshots"].pop(),
                     lambda body: body["full_vocabulary"]["snapshots"][0].update(step=1),
                     lambda body: body["full_vocabulary"]["snapshots"][0]["probability_bits"].pop(),
                     lambda body: body["measurements"][-2].update(seconds=-1))
        for number, mutate in enumerate(mutations):
            changed = copy.deepcopy(events)
            mutate(changed[-1]["observation"])
            self.assertNotEqual(json.dumps(changed), json.dumps(events))
            path = self.root / (f"invalid-probe-{index}-{number}.jsonl")
            path.write_text("\n".join(map(json.dumps, changed)) + "\n")
            self.command(f"reject-probe-{index}-{number}", ["score", "inspect", *flags({**selected,
                         "log": path, "exit-code": 0})], success=False)

    def test_steps_are_validated_before_execution(self):
        identity = self.targets[0][2]
        selected = self.inputs(identity, call=60)
        planned = self.command("probe-plan", ["score", "plan", *flags({**selected, "probe-steps": "[0]"})])
        envelope = json.loads(planned.stdout)
        _, _, accepted = scoring.decode(envelope)
        self.assertEqual(accepted.steps, (0,))
        for index, steps in enumerate(([], [0, 0], [1, 0], [-1], [False], [0.5], [len(self.source_result["behavior_bits"])], None)):
            self.command("invalid-probe-plan-" + str(index), ["score", "plan", *flags({**selected,
                         "probe-steps": json.dumps(steps)})], success=False)
            with self.assertRaises(ValueError):
                scoring.decode({**envelope, "probe_steps": steps})

    def test_missing_or_wrong_permission_prevents_probe_execution(self):
        cache, checkpoint, identity = self.targets[0]
        planned = self.command("probe-permission-plan", ["score", "plan", *flags({**self.inputs(identity, call=61), "probe-steps": "[0]"})])
        call = json.loads(planned.stdout)
        for permission in ("", json.dumps({"binding": call["binding"], "program": "incorrect"}) + "\n"):
            output = io.StringIO()
            with self.assertRaises(ValueError):
                scoring.run(SimpleNamespace(cache=cache, adapter=checkpoint, config=self.config),
                            loader=load, source=io.StringIO(planned.stdout + permission), output=output)
            stages = [json.loads(line)["stage"] for line in output.getvalue().splitlines()]
            self.assertEqual(stages[-1], "consumed")
            self.assertNotIn("cross_score", stages)
            self.assertNotIn("score_result", stages)


if __name__ == "__main__":
    unittest.main()
