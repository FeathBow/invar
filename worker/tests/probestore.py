from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from worker.distribution import FP32_BYTES, Probe, Snapshot
from worker.probeoutput import Rows, chunks, encoded, write
from worker.probestore import Store
from worker.resident import Transcript


class StorageTests(unittest.TestCase):
    def test_exact_words_inventory_and_repeated_reads(self):
        snapshots = (Snapshot(step=0, probability_bits=(0x80000000, 1, 0x3f800000)),
                     Snapshot(step=3, probability_bits=(0x3e800000, 0x3f400000, 0)))
        with tempfile.TemporaryFile() as stream:
            stored = Store(stream, probe=Probe(steps=(0, 3)))
            for snapshot in snapshots:
                stored.append(snapshot)
            actual = stored.completed()
            self.assertEqual(actual.vocabulary, 3)
            self.assertEqual(tuple(actual.snapshots), snapshots)
            self.assertEqual(tuple(actual.snapshots), snapshots)
            self.assertIs(stored.completed(), actual)
            self.assertEqual(stream.seek(0, 2), len(snapshots) * 3 * FP32_BYTES)
            with self.assertRaisesRegex(RuntimeError, "complete"):
                stored.append(snapshots[-1])

    def test_missing_reordered_repeated_or_different_width_cannot_complete(self):
        first = Snapshot(step=0, probability_bits=(0x3f000000, 0x3f000000))
        for invalid in (first, Snapshot(step=2, probability_bits=(0x3f800000,)),
                        Snapshot(step=3, probability_bits=first.probability_bits)):
            with self.subTest(invalid=invalid), tempfile.TemporaryFile() as stream:
                stored = Store(stream, probe=Probe(steps=(0, 2)))
                stored.append(first)
                with self.assertRaises(ValueError):
                    stored.append(invalid)
                with self.assertRaisesRegex(ValueError, "complete selected"):
                    stored.completed()
        with tempfile.TemporaryFile() as stream:
            stored = Store(stream, probe=Probe(steps=(0, 2)))
            with self.assertRaisesRegex(ValueError, "in order"):
                stored.append(Snapshot(step=2, probability_bits=first.probability_bits))

    def test_corruption_truncation_and_extra_bytes_are_not_reported_as_original(self):
        original = Snapshot(step=0, probability_bits=(0x3f000000, 0x3f000000))
        for change in ("corruption", "truncation", "extension"):
            with self.subTest(change=change), tempfile.TemporaryFile() as stream:
                stored = Store(stream, probe=Probe(steps=(0,)))
                stored.append(original)
                observed = stored.completed()
                damage(stream, change)
                with self.assertRaisesRegex(ValueError, "changed after capture"):
                    tuple(observed.snapshots)

    def test_empty_ownership_closed_file_and_readonly_io_fail(self):
        with tempfile.TemporaryFile() as stream:
            stream.write(b"existing")
            with self.assertRaisesRegex(ValueError, "empty owned"):
                Store(stream, probe=Probe(steps=(0,)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readonly"
            path.touch()
            with path.open("rb") as stream:
                stored = Store(stream, probe=Probe(steps=(0,)))
                with self.assertRaises(io.UnsupportedOperation):
                    stored.append(Snapshot(step=0, probability_bits=(0x3f800000,)))
        with tempfile.TemporaryFile() as stream:
            stored = Store(stream, probe=Probe(steps=(0,)))
            stored.append(Snapshot(step=0, probability_bits=(0x3f800000,)))
            observed = stored.completed()
        with self.assertRaises(ValueError):
            tuple(observed.snapshots)


def damage(stream, change):
    if change == "corruption":
        stream.seek(0)
        stream.write(b"\x01")
    elif change == "truncation":
        stream.truncate(1)
    else:
        stream.seek(0, 2)
        stream.write(b"extra")
    stream.flush()


class OutputTests(unittest.TestCase):
    def test_incremental_output_matches_existing_json_bytes(self):
        snapshots = (Snapshot(step=0, probability_bits=(0x80000000, 0x3f800000)),
                     Snapshot(step=2, probability_bits=(0x3f000000, 0x3f000000)))
        with tempfile.TemporaryFile() as stream:
            stored = Store(stream, probe=Probe(steps=(0, 2)))
            for snapshot in snapshots:
                stored.append(snapshot)
            observed = stored.completed()
            common = {"text": '中文😀 \\"\n', "number": 0.5, "zero": -0.0, "flag": False, "none": None, "values": [1, 2]}
            value = {**common, "nested": {"snapshots": Rows(snapshots=observed.snapshots)}}
            expected = json.dumps({**common, "nested": {"snapshots": [asdict(item) for item in snapshots]}}, allow_nan=False) + "\n"
            self.assertEqual("".join(encoded(value)), expected)
            output = io.StringIO()
            write(value, output=output)
            self.assertEqual(output.getvalue(), expected)

    def test_storage_failure_cannot_emit_a_complete_json_record(self):
        with tempfile.TemporaryFile() as stream:
            stored = Store(stream, probe=Probe(steps=(0,)))
            stored.append(Snapshot(step=0, probability_bits=(0x3f800000,)))
            observed = stored.completed()
            stream.truncate(1)
            output = io.StringIO()
            with self.assertRaises(ValueError):
                write({"snapshots": Rows(snapshots=observed.snapshots)}, output=output)
            self.assertFalse(output.getvalue().endswith("\n"))
            with self.assertRaises(json.JSONDecodeError):
                json.loads(output.getvalue())

    def test_transcript_owns_exact_framing_and_digest_for_both_emitters(self):
        snapshot = Snapshot(step=0, probability_bits=(0x3f800000,))
        output = io.StringIO()
        transcript = Transcript(output)
        transcript.emit("before", {"binding": 7})
        transcript.emit_stream("score_result", {"snapshots": Rows(snapshots=(snapshot,))}, encode=chunks)
        expected = json.dumps({"stage": "before", "binding": 7}) + "\n"
        expected += json.dumps({"stage": "score_result", "snapshots": [asdict(snapshot)]}) + "\n"
        self.assertEqual(output.getvalue(), expected)
        self.assertEqual(transcript.digest.hexdigest(), hashlib.sha256(expected.encode()).hexdigest())

    def test_failed_probe_does_not_complete_its_transcript_frame(self):
        with tempfile.TemporaryFile() as stream:
            stored = Store(stream, probe=Probe(steps=(0,)))
            stored.append(Snapshot(step=0, probability_bits=(0x3f800000,)))
            observed = stored.completed()
            stream.truncate(1)
            output = io.StringIO()
            transcript = Transcript(output)
            with self.assertRaises(ValueError):
                transcript.emit_stream("score_result", {"snapshots": Rows(snapshots=observed.snapshots)}, encode=chunks)
            self.assertFalse(output.getvalue().endswith("\n"))
            self.assertEqual(transcript.digest.hexdigest(), hashlib.sha256(output.getvalue().encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
