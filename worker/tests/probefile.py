from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from worker import probe, probefile
from worker.tests.probefixture import artifact


def encoded(value, **options):
    return json.dumps(value, ensure_ascii=False, **options).encode()


def compare(left, right):
    with BytesIO(left) as before, BytesIO(right) as after:
        return probe.compare(probefile.index(before), probefile.index(after))


def changed(value, path, replacement):
    result = deepcopy(value)
    parent = result
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = replacement
    return result


class FileProbeTests(unittest.TestCase):
    def assert_same(self, before, after):
        expected = probe.compare(probe.read(before), probe.read(after))
        self.assertEqual(compare(before, after), expected)

    def assert_rejected(self, data):
        with self.assertRaises(ValueError):
            probe.read(data)
        with self.assertRaises(ValueError):
            compare(data, data)

    def test_both_backends_preserve_nonzero_kl_and_distinct_mass_semantics(self):
        for backend in ("mlx", "vllm"):
            left = artifact(steps=(0, 2, 5), vocabulary=4, backend=backend)
            right = deepcopy(left)
            right["full_vocabulary"]["snapshots"][1]["probability_bits"] = [0x3f000000, 0x3e800000, 0x3e000000, 0x3e000000]
            right["full_vocabulary"]["snapshots"][2]["probability_bits"][0] = 0
            if backend == "mlx":
                right["log_probability_bits"][5] = 0xff800000
            with self.subTest(backend=backend):
                self.assert_same(encoded(left), encoded(right))
                result = compare(encoded(left), encoded(right))
                self.assertEqual(result["observations"][-1]["kl_reference_candidate"], {"kind": "positive_infinity"})

    def test_member_order_whitespace_escapes_and_unicode_encodings(self):
        value = artifact(steps=(0, 1), prompt='边界 😀 \\" ] }\n')
        value = dict(reversed(list(value.items())))
        value["full_vocabulary"] = dict(reversed(list(value["full_vocabulary"].items())))
        text = "\n\t" + json.dumps(value, ensure_ascii=False, indent=2) + "\r\n"
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be"):
            data = text.encode(encoding)
            with self.subTest(encoding=encoding):
                self.assert_same(data, data)
                self.assertEqual(compare(data, data)["reference"]["artifact_sha256"], hashlib.sha256(data).hexdigest())

    def test_vector_and_quoted_metadata_cross_real_chunk_boundaries(self):
        value = artifact(steps=(0, 2), vocabulary=8192, prompt='中文😀 \\" ' * 6000)
        data = encoded(value)
        self.assert_same(data, data)

    def test_inventory_and_support_rejections_match_materialized_reader(self):
        original = artifact(steps=(0, 2), backend="mlx")
        cases = [
            (("full_vocabulary", "steps"), [2, 0]),
            (("full_vocabulary", "vocabulary"), True),
            (("full_vocabulary", "raw_payload_bytes"), 4),
            (("full_vocabulary", "snapshots"), []),
            (("full_vocabulary", "snapshots", 0, "step"), 2),
            (("full_vocabulary", "snapshots", 1, "probability_bits"), [0x3f800000]),
            (("full_vocabulary", "snapshots", 1, "probability_bits"), [0, 0]),
            (("full_vocabulary", "snapshots", 1, "probability_bits"), [0x7f800000, 0]),
            (("full_vocabulary", "snapshots", 1, "probability_bits"), [True, 0x3f800000]),
            (("full_vocabulary", "snapshots", 1, "probability_bits"), [0, 0x3f800000]),
            (("log_probability_bits",), [0xbf800000]),
            (("measurements", 0, "peak_active"), 0),
            (("target", "tokenizer"), "c" * 64),
            (("use_admission",), "accepted"),
        ]
        for path, replacement in cases:
            with self.subTest(path=path, replacement=replacement):
                self.assert_rejected(encoded(changed(original, path, replacement)))

    def test_malformed_tail_duplicates_and_nonfinite_values_never_complete(self):
        data = encoded(artifact(steps=(0, 2)))
        cases = [data[:-1], data + b"null", data.rstrip()[:-1] + b",}",
                 data.replace(b'"step": 0', b'"step": 0, "step": 0'),
                 data.replace(b'"vocabulary": 2', b'"vocabulary": 2, "vocabulary": 2'),
                 data[:-1] + b', "format": "duplicate"}',
                 data.replace(b'"seconds": 0.1', b'"seconds": NaN'),
                 data.replace(b'"snapshots": [', b'"snapshots": [,'),
                 data.replace(b'"step": 0', b'"step": 0e'),
                 data.replace(b'"probability_bits": [', b'"probability_bits": [1e999,')]
        for index, value in enumerate(cases):
            with self.subTest(case=index):
                self.assert_rejected(value)

    def test_changed_snapshot_bytes_cannot_contribute_a_result(self):
        data = encoded(artifact(steps=(0, 2)))
        with BytesIO(data) as stream:
            observed = probefile.index(stream)
            entry = observed.vectors.snapshots.entries[-1]
            stream.seek(entry.location.offset)
            stream.write(b" " * entry.location.size)
            with self.assertRaisesRegex(ValueError, "changed after indexing"):
                probe.compare(observed, observed)

    def test_source_and_step_mismatches_do_not_form_one_comparison(self):
        original = encoded(artifact(steps=(0, 2)))
        for candidate in (artifact(steps=(0, 2), prompt="other"), artifact(steps=(1, 2)), artifact(steps=(0, 2), vocabulary=4)):
            with self.subTest(candidate=candidate["full_vocabulary"]["steps"]), self.assertRaises(ValueError):
                compare(original, encoded(candidate))

    def test_command_emits_only_a_complete_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = (Path(directory) / name for name in ("reference.json", "candidate.json"))
            data = encoded(artifact(steps=(0, 2)))
            left.write_bytes(data)
            right.write_bytes(data)
            command = [sys.executable, "-B", "-m", "worker.probe", "--reference", str(left), "--candidate", str(right)]
            success = subprocess.run(command, capture_output=True, check=True)
            observed = json.loads(success.stdout)
            self.assertGreaterEqual(observed.pop("reduction_seconds"), 0)
            self.assertEqual(observed.pop("implementation"), probe.implementation())
            self.assertEqual(observed, compare(data, data))
            right.write_bytes(data[:-1])
            failure = subprocess.run(command, capture_output=True, check=False)
            self.assertNotEqual(failure.returncode, 0)
            self.assertEqual(failure.stdout, b"")


if __name__ == "__main__":
    unittest.main()
