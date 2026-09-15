import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import hashlib
import io
import json
import time
import unittest

from worker.batch import decode
from worker.resident import History, Owner, Transcript, close, closing, release
from worker.tests.hf.batch import frame, requests


def cpu_measure(stage, operation, *, emit):
    start = time.process_time()
    result = operation()
    emit(stage, {"cpu_seconds": time.process_time() - start})
    return result


class ResidentTests(unittest.TestCase):
    def setUp(self):
        self.owner = Owner(role="inference", session=2)
        self.output = io.StringIO()
        self.transcript = Transcript(self.output)
        self.calls = decode(json.dumps(frame(requests()))).calls
        self.loads = tuple(call.load for call in self.calls)
        self.transcript.emit("fixture", {"text": "Unicode: 中文", "behavior": [-0.0]})
        self.original = self.output.getvalue()
        self.request = {"format": "invar-resident-v1", "action": "release", "owner": self.owner.value(),
                        "result_sha256": hashlib.sha256(self.original.encode()).hexdigest(),
                        "loads": [{"binding": item.binding(), "program": item.program} for item in self.loads]}

    def released(self, request, operation):
        release(self.owner, self.loads, source=io.StringIO(request + "\n"), transcript=self.transcript,
                operation=operation, measure=cpu_measure)

    def test_acknowledgement_follows_release_and_retains_exact_digest_and_inventory(self):
        def operation():
            self.assertEqual(self.output.getvalue(), self.original)

        self.released(json.dumps(self.request), operation)
        actual = json.loads(self.output.getvalue().splitlines()[-1])
        measurement = actual.pop("measurement")
        self.assertEqual(actual, {"stage": "released", **{key: value for key, value in self.request.items() if key != "action"}})
        self.assertEqual(json.loads(measurement)["stage"], "released")
        self.assertGreaterEqual(json.loads(measurement)["cpu_seconds"], 0)
        self.assertTrue(measurement.endswith("\n"))
        self.assertIn("[-0.0]", self.original)

    def test_wrong_owner_digest_or_load_never_releases(self):
        def forbidden():
            self.fail("A changed acknowledgement request reached release")

        invalid = [{**self.request, "result_sha256": "0" * 64}, {**self.request, "extra": 1},
                   {**self.request, "action": "close"}, {**self.request, "loads": []},
                   {**self.request, "loads": self.request["loads"][::-1]},
                   {**self.request, "owner": {"role": "learning", "session": 2}},
                   {**self.request, "owner": {"role": "inference", "session": True}}]
        changed = copy.deepcopy(self.request)
        changed["loads"][-1]["program"] += " changed"
        invalid.append(changed)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.released(json.dumps(value), forbidden)
        duplicate = json.dumps(self.request).replace('"result_sha256":', '"result_sha256":"wrong","result_sha256":')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.released(duplicate, forbidden)
        self.assertEqual(self.output.getvalue(), self.original)

    def test_failed_release_emits_no_acknowledgement(self):
        def failed():
            raise RuntimeError("Actual release failed")

        with self.assertRaisesRegex(RuntimeError, "Actual release failed"):
            self.released(json.dumps(self.request), failed)
        self.assertEqual(self.output.getvalue(), self.original)

    def test_shutdown_is_explicit_and_measured_before_clean_eof(self):
        request = {"format": "invar-resident-v1", "owner": self.owner.value(), "action": "close"}
        closing(self.owner, request)
        with self.assertRaises(ValueError):
            closing(self.owner, {**request, "groups": 0})
        closed = []
        close(self.owner, 0, source=io.StringIO(), transcript=self.transcript,
              operation=lambda: closed.append(True), measure=cpu_measure)
        self.assertEqual(closed, [True])
        actual = json.loads(self.output.getvalue().splitlines()[-1])
        self.assertEqual((actual["stage"], actual["groups"]), ("closed", 0))
        self.assertEqual(json.loads(actual["measurement"])["stage"], "closed")
        with self.assertRaisesRegex(ValueError, "follows"):
            close(self.owner, 0, source=io.StringIO("extra\n"), transcript=self.transcript,
                  operation=lambda: closed.append(True), measure=cpu_measure)

    def test_history_keeps_all_three_freshness_domains_after_release(self):
        initial = History()
        history = initial.advance(self.calls)
        self.assertEqual((initial.groups, history.groups), (0, 1))
        for name in ("call", "attempt", "instance"):
            values = requests()
            for value in values:
                binding = {key: item + 100 for key, item in value["binding"].items()}
                value["binding"] = binding
                value["load"]["binding"] = binding
            values[-1]["binding"][name] = getattr(self.calls[-1].invocation, name)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "historical"):
                history.advance(decode(json.dumps(frame(values))).calls)


if __name__ == "__main__":
    unittest.main()
