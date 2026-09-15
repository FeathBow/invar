import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import hashlib
import copy
import json
import os
from dataclasses import replace
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from worker import redo
from worker.tests.hf.cohort import request as cohort_request

BINDINGS = ({"call": 2, "attempt": 2, "instance": 2}, {"call": 5, "attempt": 5, "instance": 5})
PROGRAM = "(program 1)"
POLICIES = ("a" * 64, "1" * 64, "2" * 64)
LEARNERS = ("b" * 64, "3" * 64, "4" * 64)
WORKER = """
import hashlib, json, os, struct, sys
from pathlib import Path
call = json.loads(sys.stdin.readline())
output = Path(sys.argv[sys.argv.index("--output") + 1] if "--output" in sys.argv else [a for a in sys.argv if a.startswith("--output=")][0][9:])
output.mkdir()
binding = call["invocation"]["binding"]
print(json.dumps({"stage": "load", "seconds": 0.5, "peak_allocated": 1, "peak_reserved": 2}), flush=True)
print(json.dumps({"stage": "loaded_learner", "binding": binding}), flush=True)
print(json.dumps({"stage": "probability_roles", "seconds": 0.1, "peak_allocated": 1, "peak_reserved": 2}), flush=True)
print(json.dumps({"stage": "consumed", "binding": binding, "program": call["invocation"]["program"], "request": call["request"], "load": call["load"]}), flush=True)
approval = json.loads(sys.stdin.readline())
assert approval == call["invocation"]
print(json.dumps({"stage": "reward_update", "seconds": 0.2, "peak_allocated": 1, "peak_reserved": 2}), flush=True)
metadata = {"binding": json.dumps(binding, sort_keys=True), "kind": "objective"}
if os.environ.get("REDO_ORDER", "a") == "b":
    metadata = {"kind": "objective", "binding": json.dumps(binding, sort_keys=True)}
header = json.dumps({"__metadata__": metadata, "reward/x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
payload = struct.pack("<f", float(os.environ.get("REDO_VALUE", "1.5")))
(output / "gradients.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
(output / "probabilities.json").write_bytes(b'{"roles": "fixture"}')
digests = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("gradients.safetensors", "probabilities.json")}
if os.environ.get("REDO_MISREPORT"):
    digests["gradients.safetensors"] = "0" * 64
policy, learner = ("1" * 64, "3" * 64) if binding["call"] == 2 else ("2" * 64, "4" * 64)
summary = {"loss": 1.0, "gradient_norm": 1.5, "reward_gradient_norm": 1.5, "active_tokens": 2,
           "before": call["request"]["policy"], "after": policy, "nonzero_advantages": 2}
print(json.dumps({"stage": "result", "binding": binding, "request": call["request"], "update": summary, "gradients": digests["gradients.safetensors"], "probabilities": digests["probabilities.json"], "adapter": policy, "learner": learner, "storage": "staged; not published"}), flush=True)
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(published, *, cycles=False):
    rows = []
    for index, binding in enumerate(BINDINGS):
        request = {**cohort_request(), "policy": POLICIES[index], "learner": LEARNERS[index]}
        rows.append({"stage": "consumed", "binding": binding, "program": PROGRAM, "request": request, "load": {"binding": binding, "program": "(load)"}})
        summary = {"loss": 1.0, "gradient_norm": 1.5, "reward_gradient_norm": 1.5, "active_tokens": 2,
                   "before": POLICIES[index], "after": POLICIES[index + 1], "nonzero_advantages": 2}
        rows.append({"stage": "result", "binding": binding, "request": request, "update": summary,
                     "gradients": digest(published[index] / "gradients.safetensors"), "probabilities": digest(published[index] / "probabilities.json"),
                     "adapter": POLICIES[index + 1], "learner": LEARNERS[index + 1], "storage": "staged; not published"})
        rows.append({"phase": "published", "binding": binding, "checkpoint": str(published[index]), "policy": POLICIES[index + 1], "learner": LEARNERS[index + 1], "publication": "reference"})
        if cycles:
            rows.append({"phase": "cycle", "index": index, "seconds": 1.0, "sessions": 1})
    return rows


def write(log, rows):
    log.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode())


def stage(directory, binding, value=1.5):
    directory.mkdir(parents=True)
    header = json.dumps({"__metadata__": {"binding": json.dumps(binding, sort_keys=True), "kind": "objective"},
                         "reward/x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (directory / "gradients.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", value))
    (directory / "probabilities.json").write_bytes(b'{"roles": "fixture"}')


class RedoTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="invar-redo-test-"))
        self.worker = self.root / "worker.py"
        self.worker.write_text(WORKER)
        self.initial = self.root / "initial"
        self.initial.mkdir()
        self.published = [self.root / "loop" / f"generation{index + 1}" for index in range(2)]
        for index, directory in enumerate(self.published):
            stage(directory, BINDINGS[index])
        self.log = self.root / "train.stdout"
        write(self.log, rows_for(self.published))

    def options(self, name, *, updates=2, exit_code=0):
        return redo.Options(python=sys.executable, worker=self.worker, cache=self.root / "cache", initial=self.initial,
                            reference=self.initial / "adapter.safetensors", log=self.log, output=self.root / name,
                            updates=updates, exit_code=exit_code)

    def run_replay(self, name, *, options=None, **environment):
        def run(command, **keywords):
            return subprocess.run(command, env={**os.environ, **environment}, **keywords)
        return redo.run(options or self.options(name), redo.Services(run=run, clock=time.perf_counter))

    def test_identical_replays_are_equal_even_with_different_metadata_order(self):
        report = self.run_replay("equal", REDO_ORDER="b")
        rows = [json.loads(line) for line in (self.root / "equal" / "calls.jsonl").read_text().splitlines()]
        self.assertEqual((report["updates"], report["equal_results"], report["terminal"]), (2, 2, "publication count and exit status"))
        self.assertEqual([row["input_checkpoint"] for row in rows], [str(self.initial), str(self.published[0])])
        self.assertTrue(all(row["equal_fields"]["gradients"] for row in rows))
        self.assertFalse(any(row["gradients_file_digest_equal"] for row in rows))
        self.assertEqual(sorted(rows[0]["measured"]), ["load", "probability_roles", "reward_update"])

    def test_changed_gradient_values_are_reported_as_unequal(self):
        report = self.run_replay("changed", REDO_VALUE="2.5")
        rows = [json.loads(line) for line in (self.root / "changed" / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(report["equal_results"], 0)
        self.assertTrue(all(not row["equal_fields"]["gradients"] and row["equal_fields"]["adapter"] for row in rows))

    def test_selected_core_is_required_before_creating_the_output(self):
        executable = shutil.which("invar")
        self.assertIsNotNone(executable, "Build invar and add it to this test process's PATH")
        report = self.run_replay("selected", options=replace(self.options("selected"), core=executable))
        self.assertEqual(report["equal_results"], 2)
        with self.assertRaises(FileNotFoundError):
            self.run_replay("unavailable", options=replace(self.options("unavailable"), core=str(self.root / "missing-invar")))
        self.assertFalse((self.root / "unavailable").exists())

    def test_reused_or_reordered_reference_records_cannot_start_replays(self):
        complete = rows_for(self.published)
        duplicate = copy.deepcopy(complete)
        duplicate.insert(2, copy.deepcopy(duplicate[1]))
        reordered = [complete[1], complete[0], *complete[2:]]
        changed = copy.deepcopy(complete)
        for index in (3, 4):
            changed[index]["request"]["learner"] = "c" * 64
        for index, rows in enumerate((duplicate, reordered, changed)):
            write(self.log, rows)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.run_replay(f"invalid-{index}")
            self.assertFalse((self.root / f"invalid-{index}").exists())

    def test_nonfinite_gradient_bytes_do_not_become_a_comparison_result(self):
        with self.assertRaisesRegex(ValueError, "non-finite FP32"):
            self.run_replay("nonfinite", REDO_VALUE="nan")
        self.assertFalse((self.root / "nonfinite" / "complete.json").exists())

    def test_missing_input_checkpoint_is_rejected_before_any_worker_starts(self):
        for name in ("gradients.safetensors", "probabilities.json"):
            (self.published[0] / name).unlink()
        self.published[0].rmdir()
        with self.assertRaises(ValueError):
            self.run_replay("missing")
        self.assertFalse((self.root / "missing").exists())

    def test_reported_digests_must_match_the_files_on_both_sides(self):
        rows = rows_for(self.published)
        rows[1] = {**rows[1], "gradients": "0" * 64}
        write(self.log, rows)
        with self.assertRaisesRegex(ValueError, "Reference gradients file differs"):
            self.run_replay("reference")
        write(self.log, rows_for(self.published))
        with self.assertRaisesRegex(ValueError, "Replayed gradients file differs"):
            self.run_replay("misreported", REDO_MISREPORT="1")
        self.assertFalse((self.root / "misreported" / "complete.json").exists())

    def test_incomplete_training_stream_or_failed_process_is_rejected(self):
        complete = rows_for(self.published)
        write(self.log, complete[:3])
        with self.assertRaises(ValueError):
            self.run_replay("prefix")
        write(self.log, complete[:4])
        with self.assertRaises(ValueError):
            self.run_replay("partial")
        write(self.log, complete)
        with self.assertRaises(ValueError):
            self.run_replay("failed", options=self.options("failed", exit_code=1))
        with self.assertRaises(ValueError):
            self.run_replay("fewer", options=self.options("fewer", updates=1))
        self.assertFalse(any((self.root / name).exists() for name in ("prefix", "partial", "failed", "fewer")))

    def test_cycle_records_must_cover_every_update_and_end_the_stream(self):
        with_cycles = rows_for(self.published, cycles=True)
        write(self.log, with_cycles)
        self.assertEqual(self.run_replay("cycles")["terminal"], "cycle records")
        invalid = (with_cycles[:-1], [*with_cycles, with_cycles[-1]],
                   [*with_cycles[:-2], with_cycles[-1], with_cycles[-2]],
                   [*with_cycles[:-1], {**with_cycles[-1], "index": 5}])
        for rows in invalid:
            write(self.log, rows)
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.run_replay("invalid")
            self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
