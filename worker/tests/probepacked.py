from dataclasses import FrozenInstanceError, asdict
import json
import pickle
import struct
import subprocess
import sys
import unittest

from worker.distribution import Probe, Snapshot
from worker.probeoutput import Rows, encoded
from worker.probepacked import Packed, Row, Snapshots, packed


def examples():
    return (Snapshot(step=0, probability_bits=(0x80000000, 1, 0x3f800000)),
            Snapshot(step=3, probability_bits=(0x3e800000, 0x3f400000, 0)))


def observation():
    return Packed(probe=Probe(steps=(0, 3)), snapshots=Snapshots(rows=tuple(map(packed, examples()))))


class PackedProbeTests(unittest.TestCase):
    def test_exact_words_endianness_and_repeated_bounded_iteration(self):
        actual = observation()
        self.assertEqual(actual.vocabulary, 3)
        self.assertEqual(tuple(actual.snapshots), examples())
        self.assertEqual(tuple(actual.snapshots), examples())
        self.assertEqual(actual.snapshots.rows[0].payload, b"\x00\x00\x00\x80\x01\x00\x00\x00\x00\x00\x80\x3f")
        self.assertEqual(sum(len(row.payload) for row in actual.snapshots.rows), 24)
        with self.assertRaises(FrozenInstanceError):
            actual.snapshots.rows[0].payload = b"other"

    def test_real_separate_python_process_preserves_complete_packet(self):
        original = observation()
        program = ("import pickle,sys; value=pickle.load(sys.stdin.buffer); value.validate(); "
                   "sys.stdout.buffer.write(pickle.dumps(value,protocol=5))")
        result = subprocess.run([sys.executable, "-B", "-c", program], input=pickle.dumps(original, protocol=5),
                                capture_output=True, timeout=15, check=True)
        actual = pickle.loads(result.stdout)
        actual.validate()
        self.assertEqual(actual, original)
        self.assertEqual(tuple(actual.snapshots), examples())

    def test_incremental_json_matches_materialized_complete_words(self):
        actual = observation()
        value = {"snapshots": Rows(snapshots=actual.snapshots), "width": actual.vocabulary}
        expected = {"snapshots": [asdict(snapshot) for snapshot in examples()], "width": 3}
        self.assertEqual("".join(encoded(value)), json.dumps(expected, allow_nan=False) + "\n")

    def test_invalid_payload_types_sizes_steps_and_words_are_rejected(self):
        for value in (b"", b"\x00", bytearray(4), memoryview(bytes(4))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Row(step=0, payload=value).decoded()
        for word in (0, 0x7fc00000, 0x7f800000, 0xff800000, 0xbf000000, 0x40000000):
            with self.subTest(word=word), self.assertRaises(ValueError):
                Row(step=0, payload=struct.pack("<I", word)).decoded()
        for step in (True, -1):
            with self.subTest(step=step), self.assertRaises(ValueError):
                Row(step=step, payload=struct.pack("<I", 0x3f800000)).decoded()

    def test_complete_inventory_and_equal_widths_are_required(self):
        first, second = tuple(map(packed, examples()))
        narrow = packed(Snapshot(step=3, probability_bits=(0x3f800000,)))
        for rows in ((), (first,), (second, first), (first, first), (first, narrow), [first, second], (first, object())):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                Packed(probe=Probe(steps=(0, 3)), snapshots=Snapshots(rows=rows))

    def test_receiver_revalidates_deserialized_malformed_packet(self):
        actual = observation()
        object.__setattr__(actual.snapshots.rows[1], "payload", struct.pack("<III", 0x3f800000, 0x7fc00000, 0))
        received = pickle.loads(pickle.dumps(actual, protocol=5))
        with self.assertRaises(ValueError):
            received.validate()
        with self.assertRaises(ValueError):
            tuple(received.snapshots)

    def test_packet_equality_checks_every_word_and_step(self):
        original = observation()
        changed = Packed(probe=Probe(steps=(0, 3)), snapshots=Snapshots(rows=(original.snapshots.rows[0],
                         packed(Snapshot(step=3, probability_bits=(0x3e800000, 0x3f000000, 0x3e800000))))))
        self.assertNotEqual(original, changed)
        self.assertEqual(original, observation())

    def test_receiver_revalidates_the_probe_declaration(self):
        actual = observation()
        object.__setattr__(actual.probe, "steps", (0, 0))
        object.__setattr__(actual.snapshots.rows[1], "step", 0)
        received = pickle.loads(pickle.dumps(actual, protocol=5))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            received.validate()


if __name__ == "__main__":
    unittest.main()
