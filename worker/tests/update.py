import tempfile
import unittest
from pathlib import Path

from .cohort import request
from update import decode, snapshot


class UpdateTests(unittest.TestCase):
    def test_snapshot_binds_the_bytes_not_a_later_path_read(self):
        path = Path(tempfile.mkdtemp(prefix="invar-update-")) / "bytes"
        path.write_bytes(b"abc")
        digest, encoded = snapshot(path)
        path.write_bytes(b"changed")
        self.assertEqual(encoded, b"abc")
        self.assertEqual(digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_update_retains_the_call_and_numerical_inputs(self):
        value = {"invocation": {"binding": {"call": 7, "attempt": 11, "instance": 13},
                                "program": "checked program\n"}, "request": request(),
                 "load": {"binding": {"call": 7, "attempt": 11, "instance": 13}, "program": "load program"}}
        parsed = decode(value)
        value["request"]["samples"][0]["behavior_bits"][0] = 0
        self.assertEqual(parsed.invocation.binding(), {"call": 7, "attempt": 11, "instance": 13})
        self.assertEqual(parsed.invocation.program, "checked program\n")
        self.assertEqual(parsed.load.binding(), parsed.invocation.binding())
        self.assertEqual(parsed.load.program, "load program")
        self.assertEqual(parsed.request.samples[0].behavior_bits, (0xBF800000,))

    def test_update_requires_both_binding_and_valid_request(self):
        binding = {"binding": {"call": 7, "attempt": 11, "instance": 13}, "program": "p"}
        for value in (request(), {"request": request()}, {"invocation": binding},
                      {"invocation": binding, "request": request()},
                      {"invocation": binding, "request": request(), "load": binding, "extra": 1},
                      {"invocation": {**binding, "program": ""}, "request": request(), "load": binding},
                      {"invocation": binding, "request": {**request(), "order": []}, "load": binding}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode(value)

    def test_load_requires_its_program_and_matching_correlation(self):
        bound = {"call": 7, "attempt": 11, "instance": 13}
        invocation = {"binding": bound, "program": "update"}
        invalid = [{"binding": bound, "program": ""},
                   *({"binding": {**bound, name: 99}, "program": "load"} for name in bound)]
        for loading in invalid:
            with self.subTest(loading=loading), self.assertRaises(ValueError):
                decode({"invocation": invocation, "request": request(), "load": loading})


if __name__ == "__main__":
    unittest.main()
