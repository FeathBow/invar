import json
import math
import sys
import unittest

from worker import probe
from worker.mlx import scoring
from worker.tests.mlx.scoring import BoundScoreFixture, SCORE_ENTRY, flags
from worker.tests.probe import RELATIVE_CHECK_TOLERANCE, decimal_kl


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
            artifacts.append(probe.read(encoded))
            if index == 0:
                self.assertEqual(observed["log_probability_bits"], self.source_result["behavior_bits"])
        compared = probe.compare(*artifacts)
        for row, left, right in zip(compared["observations"], artifacts[0].vectors.snapshots,
                                   artifacts[1].vectors.snapshots, strict=True):
            for name, p, q in (("kl_reference_candidate", left, right), ("kl_candidate_reference", right, left)):
                expected = decimal_kl(p.probability_bits, q.probability_bits)
                self.assertTrue(math.isclose(row[name]["value"], float(expected),
                                rel_tol=RELATIVE_CHECK_TOLERANCE, abs_tol=0), (name, row, expected))

    def test_steps_are_validated_before_execution(self):
        identity = self.targets[0][2]
        selected = self.inputs(identity, call=60)
        planned = self.command("probe-plan", ["score", "plan", *flags({**selected, "probe-steps": "[0]"})])
        envelope = json.loads(planned.stdout)
        _, _, accepted = scoring.decode(envelope)
        self.assertEqual(accepted.steps, (0,))
        for steps in ([], [0, 0], [1, 0], [-1], [False], [0.5], [len(self.source_result["behavior_bits"])], None):
            with self.assertRaises(ValueError):
                scoring.decode({**envelope, "probe_steps": steps})


if __name__ == "__main__":
    unittest.main()
