import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import tempfile
import unittest
from pathlib import Path

from worker.tests.hf.cohort import request
from worker.update import consumed, decode, snapshot


class UpdateTests(unittest.TestCase):
    def test_consumed_advantage_comes_from_actual_batch(self):
        from worker.advantage import check
        from worker.cohort import decode as cohort
        from worker.hf.learning import Batch, Sample
        from worker.hf.objective import Profile
        from worker.hf.step import trajectory

        numerical = cohort(request())
        prepared = check(numerical)
        trajectories = tuple(trajectory(item) for item in numerical.samples)
        logical = Batch(samples=tuple(Sample(trajectory=item, proximal=item.behavior,
                                             reference=item.behavior, advantage=1.0) for item in trajectories),
                        order=numerical.order, profile=Profile(epsilon=numerical.epsilon, penalty=numerical.penalty))
        loaded = {"base": numerical.base, "assembly": numerical.assembly}
        actual = consumed(numerical, batch=logical, rewards=prepared.rewards, loaded=loaded)
        self.assertEqual([item["advantage_bits"] for item in actual["samples"]], [0x3F800000] * 2)
        self.assertNotEqual(actual["samples"][0]["advantage_bits"], numerical.samples[0].advantage_bits)
        self.assertEqual(actual["behavior_model"], request()["behavior_model"])
        self.assertNotEqual(actual["behavior_model"], loaded)
        self.assertEqual([item["behavior_bits"] for item in actual["samples"]],
                         [list(item.behavior_bits) for item in numerical.samples])

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
