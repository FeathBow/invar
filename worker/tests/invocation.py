import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from worker.invocation import decode, request


class InvocationTests(unittest.TestCase):
    def test_binding_retains_all_identities_and_exact_program(self):
        value = {"binding": {"call": 7, "attempt": 11, "instance": 13}, "program": "program\n"}
        invocation = decode(value)
        value["binding"]["attempt"] = 99
        self.assertEqual(invocation.binding(), {"call": 7, "attempt": 11, "instance": 13})
        self.assertEqual(invocation.program, "program\n")

    def test_malformed_binding_has_no_inferred_identity(self):
        for field in ("call", "attempt", "instance"):
            for value in (-1, True, "7", 7.0, None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    decode({"binding": {"call": 7, "attempt": 11, "instance": 13, field: value},
                            "program": "program"})
        for value in ({}, {"binding": {}}, {"binding": {"call": 7, "attempt": 11}, "program": "p"},
                      {"binding": {"call": 7, "attempt": 11, "instance": 13}, "program": ""}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode(value)

    def test_consumption_reports_the_request_fields_used_by_generation(self):
        value = SimpleNamespace(prompt="--seed=0\nλ", seed=-1, temperature=1.25, limit=32)
        self.assertEqual(request(value), {"prompt": "--seed=0\nλ", "seed": -1,
                                          "temperature": 1.25, "tokens": 32})

    def test_permission_requires_the_exact_consumed_invocation(self):
        initial = {"binding": {"call": 7, "attempt": 11, "instance": 13}, "program": "program\n"}
        variants = [initial, None, {**initial, "program": "other"}]
        variants += [{**initial, "binding": {**initial["binding"], field: 99}}
                     for field in ("call", "attempt", "instance")]
        code = "from worker.invocation import read, approve; value = read(); approve(value); print('approved')"
        for index, permission in enumerate(variants):
            with self.subTest(permission=permission):
                encoded = json.dumps(initial) + "\n"
                if permission is not None:
                    encoded += json.dumps(permission) + "\n"
                outcome = subprocess.run([sys.executable, "-B", "-c", code], input=encoded,
                                         cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, timeout=10)
                self.assertEqual(outcome.returncode == 0, index == 0)
                self.assertEqual(outcome.stdout, "approved\n" if index == 0 else "")


if __name__ == "__main__":
    unittest.main()
