import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace

from worker import direct
from worker import performance
from worker.tests.fixtures import resident as fixture

OWNER_COUNTS = (1, 2, 3)
REPETITIONS = 2
COHORTS = 3
POLICY = "a" * 64
IDENTITIES = {name: character * 64 for name, character in zip(("tokenizer", "base", "assembly"), "bcd", strict=True)}


def workload():
    tasks = [{"name": name, "group": "question", "prompt": "Protocol fixture 中文", "seed": index,
              "tokens": 2, "temperature": 0.5, "answer": "#### 12"} for index, name in enumerate(("first", "second"))]
    return [{"tasks": tasks, "order": [1, 0], "delivery": [0, 1]} for _ in range(COHORTS)]


def reference(options, *, count):
    command = [options.core, "evaluate", f"--python={options.python}", f"--worker={options.worker}",
               f"--cache={options.cache}", f"--adapter={options.adapter}", f"--policy={options.policy}",
               f"--worker-config={options.worker_config}", "--worker-mode=resident", "--devices=" + ",".join(map(str, range(count))),
               *(f"--{name}-digest={value}" for name, value in IDENTITIES.items())]
    with options.reference_log.open("x") as stdout:
        completed = subprocess.run(command, input=options.tasks.read_text(), text=True, stdout=stdout, capture_output=False,
                                   stderr=subprocess.PIPE, check=True)
    options.reference_log.with_suffix(".stderr.log").write_text(completed.stderr)


def manifest(options, paths):
    rows = []
    for index, current in enumerate(paths):
        rows.extend(({"name": f"invar-{index}", "route": "invar", "path": str(current.reference_log),
                      "exit_code": 0, "elapsed_seconds": 100.0},
                     {"name": f"direct-{index}", "route": "direct", "path": str(current.output),
                      "exit_code": 0, "elapsed_seconds": 100.0}))
    path = options.output.parent / "manifest.json"
    path.write_text(json.dumps({"reference_log": str(options.reference_log), "reference_exit_code": 0, "runs": rows}))
    return path


class ResidentDirectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="invar-resident-direct-"))
        cls.services = direct.Services(run=subprocess.run, spawn=subprocess.Popen, clock=time.perf_counter,
                                       environment=dict(os.environ))
        cls.paths, cls.reports = {}, {}
        for count in OWNER_COUNTS:
            root = cls.root / f"owners-{count}"
            root.mkdir()
            script = root / "worker.py"
            script.write_text(f"import sys\nsys.path.insert(0, {str(Path(direct.__file__).resolve().parents[1])!r})\n"
                              "from worker.tests.fixtures.resident import main\nmain()\n")
            configuration = root / "configuration.json"
            configuration.write_text(json.dumps({"fault": None}))
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps(workload(), ensure_ascii=False))
            cls.paths[count] = []
            for index in range(REPETITIONS):
                options = direct.Options(python=sys.executable, core=shutil.which("invar"), worker=script, cache=root,
                                         adapter=root / "protocol-adapter", policy=POLICY, tasks=tasks,
                                         reference_log=root / f"reference-{index}.jsonl", reference_exit_code=0,
                                         output=root / f"direct-{index}", mode="resident", worker_config=configuration,
                                         devices=tuple(map(str, range(count))))
                reference(options, count=count)
                cls.reports[count, index] = direct.run(options, cls.services)
                cls.paths[count].append(options)

    def report(self, count):
        options = self.paths[count][0]
        return performance.report(manifest(options, self.paths[count]), options.tasks, options.policy)

    def test_physical_owners_keep_original_groups_profiles_and_final_close(self):
        for count in OWNER_COUNTS:
            options = self.paths[count][0]
            report = self.reports[count, 0]
            self.assertEqual((report["calls"], report["equal_results"], report["loads"]), (6, 6, min(count, 2)))
            self.assertEqual([row["owner"] for row in report["close_intervals"]], list(reversed(range(count))))
            pids = []
            for owner in range(count):
                prefix = options.output / f"owner-{owner:04d}"
                status = json.loads(prefix.with_suffix(".status.json").read_text())
                child = json.loads(prefix.with_suffix(".stderr.log").read_text())
                rows = [json.loads(raw) for raw in prefix.with_suffix(".stdout.jsonl").read_text().splitlines()]
                pids.append(status["pid"])
                self.assertEqual((child["pid"], child["owner"], child["device"], child["adapter_argument"]),
                                 (status["pid"], owner, str(owner), False))
                self.assertEqual(status["process_seconds"], status["end_offset_seconds"] - status["start_offset_seconds"])
                self.assertEqual(rows[-1]["groups"], COHORTS if owner < 2 else 0)
                self.assertEqual(sum(row["stage"] == "load" for row in rows), int(owner < 2))
                self.assertEqual(sum(row["stage"] == "activation" for row in rows), COHORTS - 1 if owner < 2 else 0)
                self.assertEqual(status["stdout_sha256"], hashlib.sha256(prefix.with_suffix(".stdout.jsonl").read_bytes()).hexdigest())
            self.assertEqual(len(set(pids)), count)

    def test_complete_measurements_match_owner_partition_and_all_actual_cost_records(self):
        for count in OWNER_COUNTS:
            report = self.report(count)
            self.assertTrue(report["comparison"]["all_results_equal_to_reference"])
            expected = (fixture.TIMES["load"] + COHORTS * (fixture.TIMES["inference"] + fixture.TIMES["released"])
                        + (COHORTS - 1) * fixture.TIMES["activation"] + count * fixture.TIMES["closed"])
            for row in report["runs"]:
                self.assertEqual(row["worker_mode"], "resident")
                self.assertEqual(row["model_loads_per_cohort"], [min(count, 2), 0, 0])
                self.assertAlmostEqual(row["worker_critical_path_seconds"], expected)
                self.assertAlmostEqual(row["inference_seconds"]["total"], min(count, 2) * COHORTS * fixture.TIMES["inference"])
                self.assertAlmostEqual(row["close_seconds"]["total"], count * fixture.TIMES["closed"])
                self.check_provenance(row, count)

    def check_provenance(self, row, count):
        if row["route"] != "direct":
            return
        options = self.paths[count][int(row["name"].split("-")[-1])]
        self.assertEqual(len(row["physical_owners"]), count)
        for actual in row["physical_owners"]:
            original = (options.output / f"owner-{actual['owner']:04d}.status.json").read_bytes()
            self.assertEqual(actual["status_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(actual["status_json"].encode(), original)

    def test_forged_counts_and_overlapping_or_truncated_host_lifetimes_are_rejected(self):
        options = self.paths[3][0]
        path = options.output / "complete.json"
        original = path.read_bytes()
        value = json.loads(original)
        overlap = copy.deepcopy(value)
        overlap["cohort_intervals"][1]["start_offset_seconds"] = overlap["cohort_intervals"][0]["start_offset_seconds"]
        invalid = [overlap, {**value, "loads": 3}, {**value, "sessions": 2}, {**value, "process_seconds": value["process_seconds"] + 1},
                   {**value, "close_intervals": value["close_intervals"][::-1]}, {**value, "wall_seconds": 0.000001}]
        try:
            for changed in invalid:
                path.write_text(json.dumps(changed))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    self.report(3)
        finally:
            path.write_bytes(original)

    def test_failed_exit_and_changed_raw_output_cannot_be_replaced_by_summary(self):
        options = self.paths[3][0]
        prefix = options.output / "owner-0002"
        status_path = prefix.with_suffix(".status.json")
        status_raw = status_path.read_bytes()
        output_path = prefix.with_suffix(".stdout.jsonl")
        output_raw = output_path.read_bytes()
        try:
            status_path.write_text(json.dumps({**json.loads(status_raw), "exit_code": fixture.FAILURE_STATUS}))
            with self.assertRaises(ValueError):
                self.report(3)
            changed = json.loads(output_raw)
            changed["groups"] = 1
            output_path.write_text(fixture.encoded(changed))
            status_path.write_text(json.dumps({**json.loads(status_raw), "stdout_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()}))
            with self.assertRaises(ValueError):
                self.report(3)
        finally:
            output_path.write_bytes(output_raw)
            status_path.write_bytes(status_raw)

    def test_bad_reference_or_missing_placement_fails_before_any_child_launch(self):
        options = self.paths[3][0]
        variants = ({"devices": None}, {"devices": ("0", "1")}, {"reference_exit_code": fixture.FAILURE_STATUS})
        for index, change in enumerate(variants):
            selected = replace(options, output=options.output.parent / f"invalid-{index}", **change)
            with self.assertRaises(ValueError):
                direct.run(selected, self.services)
            self.assertFalse(selected.output.exists())

    def test_protocol_failure_reaps_all_started_children_and_retains_failed_status(self):
        options = self.paths[3][0]
        configuration = options.worker_config.read_bytes()
        try:
            for fault in ("exit", "release", "late_exit"):
                options.worker_config.write_text(json.dumps({"fault": fault}))
                selected = replace(options, output=options.output.parent / f"failed-{fault}")
                with self.subTest(fault=fault), self.assertRaises((ValueError, BrokenPipeError, subprocess.CalledProcessError)):
                    direct.run(selected, self.services)
                self.assertFalse((selected.output / "complete.json").exists())
                statuses = [json.loads(path.read_text()) for path in sorted(selected.output.glob("*.status.json"))]
                self.assertEqual(len(statuses), 3)
                self.assertTrue(any(row["exit_code"] != 0 for row in statuses))
                self.check_reaped(statuses)
        finally:
            options.worker_config.write_bytes(configuration)

    def test_valid_numerical_differences_remain_visible_in_matched_measurements(self):
        options, observed = self.changed_output("numerical")
        self.assertEqual((observed["calls"], observed["equal_results"], observed["response_tokens"]), (6, 0, 12))
        paths = [options, self.paths[1][1]]
        report = performance.report(manifest(options, paths), options.tasks, options.policy)
        self.assertFalse(report["comparison"]["all_results_equal_to_reference"])

    def test_changed_physical_profile_cannot_pass_matched_measurement_admission(self):
        options, observed = self.changed_output("profile")
        self.assertEqual(observed["equal_results"], 6)
        paths = [options, self.paths[1][1]]
        with self.assertRaisesRegex(ValueError, "profile"):
            performance.report(manifest(options, paths), options.tasks, options.policy)

    def changed_output(self, fault):
        options = self.paths[1][0]
        configuration = options.worker_config.read_bytes()
        try:
            options.worker_config.write_text(json.dumps({"fault": fault}))
            selected = replace(options, output=options.output.parent / f"changed-{fault}")
            return selected, direct.run(selected, self.services)
        finally:
            options.worker_config.write_bytes(configuration)

    def check_reaped(self, statuses):
        for row in statuses:
            self.assertIsInstance(row["exit_code"], int)
            with self.assertRaises(ProcessLookupError):
                os.kill(row["pid"], 0)


if __name__ == "__main__":
    unittest.main()
