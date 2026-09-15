import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from worker import direct
from worker.tests.hf import direct as fixture
from worker import evaluation
from worker import performance
from worker import evidence

LOAD_SECONDS = 0.001
INFERENCE_SECONDS = 0.002
INVAR_DURATIONS = (30, 40)
DIRECT_DURATIONS = (20, 24)


def worker_rows(bound):
    loaded = {"stage": "loaded_adapter", "binding": bound.consumed["binding"],
              "requested": fixture.fixture.INITIAL_POLICY, "consumed": fixture.fixture.INITIAL_POLICY,
              "model": "protocol-fixture", "revision": "fixed-fixture", "scope": "fixture only",
              **evaluation.model_binding(bound.consumed)}
    raw = fixture.output(bound)
    raw[0]["seconds"] = LOAD_SECONDS
    raw[-2]["seconds"] = INFERENCE_SECONDS
    return [{"stage": "profile", "precision": "fixture"}, raw[0], loaded, *raw[1:]]


def reference(bindings=()):
    result = []
    for row in fixture.reference():
        if row.get("stage") == "result":
            continue
        if row.get("stage") == "consumed":
            task = row["request"]
            result.extend(worker_rows(fixture.call(task, row["binding"])))
        else:
            result.append(row)
    for row in result:
        if row.get("stage") in ("loaded_adapter", "consumed", "result") or row.get("phase") == "evaluation_complete":
            row.update(bindings)
        if row.get("phase") == "evaluation":
            for sample in row["samples"]:
                sample["response_tokens"] = len(fixture.RESPONSE)
            row["summary"]["response_tokens"] = len(row["samples"]) * len(fixture.RESPONSE)
    return result


def partitioned_reference(bindings, sessions, *, declare=True):
    rows, streams, previous, position = [], None, None, 0

    def flush():
        for stream in streams or ():
            rows.extend(stream)

    for row in reference(bindings):
        stage = row.get("stage")
        if stage in ("profile", "load"):
            continue
        if stage == "loaded_adapter":
            if streams is None:
                streams, previous, position = [[] for _ in range(sessions)], [None] * sessions, 0
            slot = position % sessions
            if not streams[slot]:
                streams[slot].extend(({"stage": "profile", "precision": "fixture"},
                                      {"stage": "load", "seconds": LOAD_SECONDS, "peak_allocated": 100, "peak_reserved": 200}))
            elif previous[slot] is not None:
                streams[slot].append({"stage": "unloaded_adapter", "binding": previous[slot], "program": "load fixture"})
            previous[slot] = row["binding"]
            position += 1
        if stage in ("loaded_adapter", "consumed", "inference", "result"):
            slot = (position - 1) % sessions
            streams[slot].append({**row, "load": {"binding": row["binding"], "program": "load fixture"}} if stage == "consumed" else row)
        elif row.get("phase") == "evaluation":
            flush()
            streams = None
            rows.append(row)
        else:
            rows.append({**row, "sessions": sessions} if declare else row)
    return rows


def batch_reference(bindings):
    return partitioned_reference(bindings, 1, declare=False)


def mismatched_summary(rows):
    row = next(row for row in rows if row.get("phase") == "evaluation")
    row["samples"][0]["response_tokens"] += 1
    row["summary"]["response_tokens"] += 1


class PerformanceChecks(unittest.TestCase):
    bindings = ()

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="invar-performance-test-"))
        cls.options = direct.Options(python=sys.executable, worker=cls.root / "worker.py", cache=cls.root,
                                     adapter=cls.root / "adapter", policy=fixture.fixture.INITIAL_POLICY,
                                     tasks=cls.root / "tasks.json", reference_log=cls.root / "reference.jsonl",
                                     reference_exit_code=0, output=cls.root / "direct-0", mode="process")
        cls.options.tasks.write_text(json.dumps(fixture.fixture.declarations()))
        fixture.fixture.write(cls.options.reference_log, reference(cls.bindings))
        _, _, calls = direct.calls(cls.options)
        cls.count = len(calls)
        payload = {bound.consumed["binding"]["call"]: worker_rows(bound) for bound in calls}
        source = "import json,sys\nfirst=json.loads(sys.stdin.readline())\nsecond=json.loads(sys.stdin.readline())\n"
        source += "assert first==second\n"
        source += f"payload={payload!r}\nrecords=payload[first['binding']['call']]\n"
        source += "for row in records: print(json.dumps(row),flush=True)\n"
        cls.options.worker.write_text(source)
        cls.runs = []
        cls.prepare_runs(direct.Services(run=subprocess.run, clock=time.perf_counter))

    @classmethod
    def prepare_runs(cls, services):
        for index, elapsed in enumerate(INVAR_DURATIONS):
            path = cls.root / f"invar-{index}.jsonl"
            fixture.fixture.write(path, reference(cls.bindings))
            cls.runs.append({"name": f"invar-{index}", "route": "invar", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})
        for index, elapsed in enumerate(DIRECT_DURATIONS):
            path = cls.root / f"direct-{index}"
            direct.run(replace(cls.options, output=path), services)
            cls.runs.append({"name": f"direct-{index}", "route": "direct", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})

    def report(self, runs=None, *, core_executable="invar"):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"reference_log": str(self.options.reference_log), "reference_exit_code": 0,
                                    "runs": self.runs if runs is None else runs}))
        return performance.report(path, self.options.tasks, self.options.policy, core_executable=core_executable)

    def measurements(self, rows, name="measured-reference.jsonl"):
        path = self.root / name
        fixture.fixture.write(path, rows)
        return evidence.measurements(path, self.options.tasks, self.options.policy, exit_code=0)

    def test_selected_core_is_required_for_the_complete_report(self):
        selected = shutil.which("invar")
        self.assertIsNotNone(selected)
        self.assertTrue(self.report(core_executable=selected)["comparison"]["all_results_equal_to_reference"])
        with self.assertRaises(FileNotFoundError):
            self.report(core_executable=self.root / "missing-core")

    def test_repeated_routes_and_weighted_rates(self):
        result = self.report()
        self.assertTrue(result["comparison"]["all_results_equal_to_reference"])
        self.assertEqual(result["comparison"]["invar_minus_direct_elapsed_seconds"], 13)
        for row, source in zip(result["runs"], self.runs, strict=True):
            self.assertEqual(row["response_tokens"], self.count * len(fixture.RESPONSE))
            self.assertEqual(row["response_tokens_per_elapsed_second"], row["response_tokens"] / source["elapsed_seconds"])
            self.assertAlmostEqual(row["inference_seconds"]["total"], self.count * INFERENCE_SECONDS)
            self.assertFalse(row["concurrent"])
            self.assertAlmostEqual(row["worker_critical_path_seconds"], self.count * (LOAD_SECONDS + INFERENCE_SECONDS))
            self.assertAlmostEqual(row["seconds_outside_worker_critical_path"],
                                   source["elapsed_seconds"] - self.count * (LOAD_SECONDS + INFERENCE_SECONDS))

    def test_failed_incomplete_or_reused_runs_rejected(self):
        invalid = (self.runs[:-1], [*self.runs, self.runs[0]],
                   [{**self.runs[0], "exit_code": 1}, *self.runs[1:]],
                   [{**self.runs[0], "elapsed_seconds": 0}, *self.runs[1:]],
                   [*self.runs, {**self.runs[0], "name": "aliased-run"}])
        for runs in invalid:
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                self.report(runs)

    def test_changed_numerical_result_is_retained(self):
        rows = reference(self.bindings)
        result = next(row for row in rows if row.get("stage") == "result")
        result["behavior_bits"][0] += 1
        word = result["behavior_bits"][0]
        result["behavior"][0] = struct.unpack("!f", struct.pack("!I", word))[0]
        path = self.root / "changed-result.jsonl"
        fixture.fixture.write(path, rows)
        report = self.report([{**self.runs[0], "path": str(path)}, *self.runs[1:]])
        self.assertFalse(report["comparison"]["all_results_equal_to_reference"])
        self.assertEqual(report["runs"][0]["equal_results"], self.count - 1)

    def test_profile_and_summary_mismatch_rejected(self):
        changes = (
            lambda rows: next(row for row in rows if row.get("stage") == "profile").update(precision="different"),
            mismatched_summary,
        )
        for change in changes:
            rows = reference(self.bindings)
            change(rows)
            path = self.root / "changed-profile.jsonl"
            fixture.fixture.write(path, rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.report([{**self.runs[0], "path": str(path)}, *self.runs[1:]])

    def test_raw_direct_corruption_cannot_hide_behind_summary(self):
        original = Path(self.runs[-1]["path"])
        changed = self.root / "corrupt-direct"
        changed.mkdir()
        for path in original.iterdir():
            (changed / path.name).write_bytes(path.read_bytes())
        target = changed / "0000.stdout.jsonl"
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        rows[-1]["behavior_bits"][0] += 1
        fixture.fixture.write(target, rows)
        with self.assertRaises(ValueError):
            self.report([*self.runs[:-1], {**self.runs[-1], "path": str(changed)}])


class TokenizerPerformanceChecks(PerformanceChecks):
    bindings = (("tokenizer", fixture.TOKENIZER),)


class MaterializedPerformanceChecks(PerformanceChecks):
    bindings = (("tokenizer", fixture.TOKENIZER), ("base", fixture.BASE), ("assembly", fixture.ASSEMBLY))

    def test_missing_or_different_loaded_identity_is_rejected(self):
        for field, _ in self.bindings:
            for value in (None, "0" * 64):
                rows = reference(self.bindings)
                loaded = next(row for row in rows if row["stage"] == "loaded_adapter")
                del loaded[field]
                if value is not None:
                    loaded[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.measurements(rows)

    def test_materialization_is_part_of_the_recorded_profile_identity(self):
        expected = self.measurements(reference(self.bindings))["measurements"][0]["profile_sha256"]
        for field, _ in self.bindings:
            bindings = {**dict(self.bindings), field: "0" * 64}
            rows = reference(bindings.items())
            with self.subTest(field=field):
                self.assertNotEqual(self.measurements(rows)["measurements"][0]["profile_sha256"], expected)

    def test_concurrent_sessions_use_the_longest_session_per_cohort_as_the_critical_path(self):
        rows = partitioned_reference(self.bindings, 2)
        self.assertEqual(self.measurements(rows)["sessions_per_cohort"], [2, 2])
        reference_log = self.root / "partitioned-reference.jsonl"
        fixture.fixture.write(reference_log, rows)
        runs = []
        for index, elapsed in enumerate((0.010, 0.012)):
            path = self.root / f"partitioned-invar-{index}.jsonl"
            fixture.fixture.write(path, rows)
            runs.append({"name": f"partitioned-invar-{index}", "route": "invar", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})
        source = replace(self.options, reference_log=reference_log)
        _, _, calls = direct.calls(source)
        process = replace(self.options, reference_log=reference_log, worker=self.root / "partitionedprocess.py", mode="process")
        payload = {bound.consumed["binding"]["call"]: worker_rows(bound) for bound in calls}
        script = "import json,sys\nfirst=json.loads(sys.stdin.readline())\nsecond=json.loads(sys.stdin.readline())\n"
        script += "assert second=={key:first[key] for key in ('binding','program')}\n"
        script += f"payload={payload!r}\nrecords=payload[first['binding']['call']]\n"
        script += "for row in records: print(json.dumps(row),flush=True)\n"
        process.worker.write_text(script)
        serial = []
        for index, elapsed in enumerate(DIRECT_DURATIONS):
            path = self.root / f"partitionedprocess-{index}"
            direct.run(replace(process, output=path), direct.Services(run=subprocess.run, clock=time.perf_counter))
            serial.append({"name": f"partitionedprocess-{index}", "route": "direct", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})
        manifest = self.root / "partitioned-manifest.json"
        manifest.write_text(json.dumps({"reference_log": str(reference_log), "reference_exit_code": 0, "runs": [*runs, *serial]}))
        with self.assertRaisesRegex(ValueError, "same number of model loads per cohort"):
            performance.report(manifest, self.options.tasks, self.options.policy)
        observed = evidence.measurements(runs[0]["path"], source.tasks, source.policy, exit_code=0)
        self.assertTrue(observed["concurrent"])
        self.assertEqual(observed["sessions_per_cohort"], [2, 2])
        longest = sum(max(row["seconds"] + row["inference_seconds"] for row in observed["loads"] if row["cohort"] == index)
                      for index in range(observed["cohorts"]))
        self.assertEqual((observed["cohorts"], observed["sessions_per_cohort"]), (2, [2, 2]))
        self.assertAlmostEqual(observed["critical_path_seconds"], longest)
        cumulative = sum(row["seconds"] + row["inference_seconds"] for row in observed["loads"])
        self.assertAlmostEqual(cumulative, 4 * LOAD_SECONDS + self.count * INFERENCE_SECONDS)
        self.assertLess(longest, 0.010)
        self.assertLess(0.010, cumulative)

    def test_batch_reference_is_grouped_into_one_session_per_cohort(self):
        observed = self.measurements(batch_reference(self.bindings))
        self.assertEqual(observed["sessions_per_cohort"], [1, 1])
        self.assertEqual(sum(row["calls"] for row in observed["loads"]), self.count)
        self.assertEqual(len(self.measurements(reference(self.bindings))["loads"]), self.count)
        broken = [row for row in batch_reference(self.bindings) if row.get("stage") != "load"]
        with self.assertRaises(ValueError):
            self.measurements(broken)

    def test_batch_reference_matches_session_replays_and_rejects_mismatched_load_counts(self):
        rows = batch_reference(self.bindings)
        reference_log = self.root / "batch-reference.jsonl"
        fixture.fixture.write(reference_log, rows)
        services = direct.Services(run=subprocess.run, clock=time.perf_counter)
        session = replace(self.options, reference_log=reference_log, worker=self.root / "session.py", mode="session")
        _, _, calls = direct.calls(session)
        session.worker.write_text(fixture.session_source(calls, load_seconds=LOAD_SECONDS, inference_seconds=INFERENCE_SECONDS))
        runs = []
        for index, elapsed in enumerate(INVAR_DURATIONS):
            path = self.root / f"batch-invar-{index}.jsonl"
            fixture.fixture.write(path, rows)
            runs.append({"name": f"batch-invar-{index}", "route": "invar", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})
        for index, elapsed in enumerate(DIRECT_DURATIONS):
            path = self.root / f"session-{index}"
            direct.run(replace(session, output=path), services)
            runs.append({"name": f"session-{index}", "route": "direct", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})
        manifest = self.root / "batch-manifest.json"
        manifest.write_text(json.dumps({"reference_log": str(reference_log), "reference_exit_code": 0, "runs": runs}))
        result = performance.report(manifest, self.options.tasks, self.options.policy)
        self.assertEqual((result["comparison"]["cohorts"], result["comparison"]["sessions_per_cohort"]), (2, [1, 1]))
        self.assertTrue(result["comparison"]["all_results_equal_to_reference"])
        self.assertEqual(result["comparison"]["invar_minus_direct_elapsed_seconds"], 13)
        for row, source in zip(result["runs"], runs, strict=True):
            self.assertEqual((row["sessions_per_cohort"], row["calls"], row["concurrent"]), ([1, 1], self.count, False))
            self.assertAlmostEqual(row["seconds_outside_worker_critical_path"],
                                   source["elapsed_seconds"] - 2 * LOAD_SECONDS - self.count * INFERENCE_SECONDS)
        completion = Path(runs[-1]["path"]) / "complete.json"
        original = completion.read_bytes()
        for change in ({"concurrent": True}, {"cohorts": 1}, {"loads": 1}):
            completion.write_bytes(json.dumps({**json.loads(original), **change}).encode())
            with self.subTest(change=change), self.assertRaises(ValueError):
                performance.report(manifest, self.options.tasks, self.options.policy)
        completion.write_bytes(original)
        process = replace(session, worker=self.root / "batchprocess.py", mode="process")
        payload = {bound.consumed["binding"]["call"]: worker_rows(bound) for bound in calls}
        source = "import json,sys\nfirst=json.loads(sys.stdin.readline())\nsecond=json.loads(sys.stdin.readline())\n"
        source += "assert second=={key:first[key] for key in ('binding','program')}\n"
        source += f"payload={payload!r}\nrecords=payload[first['binding']['call']]\n"
        source += "for row in records: print(json.dumps(row),flush=True)\n"
        process.worker.write_text(source)
        mixed = list(runs[:len(INVAR_DURATIONS)])
        for index, elapsed in enumerate(DIRECT_DURATIONS):
            path = self.root / f"batchprocess-{index}"
            direct.run(replace(process, output=path), services)
            mixed.append({"name": f"batchprocess-{index}", "route": "direct", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})
        manifest.write_text(json.dumps({"reference_log": str(reference_log), "reference_exit_code": 0, "runs": mixed}))
        with self.assertRaisesRegex(ValueError, "same number of model loads per cohort"):
            performance.report(manifest, self.options.tasks, self.options.policy)


if __name__ == "__main__":
    unittest.main()
