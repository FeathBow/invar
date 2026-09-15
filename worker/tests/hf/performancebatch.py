import unittest

try:
    import torch  # noqa: F401
    import vllm  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from worker import direct
from worker import evidence
from worker import performance
from worker.tests.hf import direct as replay
from worker.tests.hf import directbatch as batch
from worker.tests.hf import performance as fixture

BINDINGS = (("tokenizer", replay.TOKENIZER), ("base", replay.BASE), ("assembly", replay.ASSEMBLY))


def reference():
    result, calls, consumed = [], [], None
    for row in fixture.batch_reference(BINDINGS):
        if row.get("stage") == "consumed":
            consumed = row
        elif row.get("stage") == "result":
            calls.append(direct.Call(consumed=consumed, result=row))
        elif row.get("phase") == "evaluation":
            result.extend(batch.output(calls, load_seconds=fixture.LOAD_SECONDS, inference_seconds=fixture.INFERENCE_SECONDS))
            result.append(row)
            calls = []
        elif "phase" in row:
            result.append(row)
    return result


class BatchMeasurementChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="invar-batch-measurement-"))
        cls.options = direct.Options(python=sys.executable, worker=cls.root / "worker.py", cache=cls.root,
                                     adapter=cls.root / "adapter", policy=replay.fixture.INITIAL_POLICY,
                                     tasks=cls.root / "tasks.json", reference_log=cls.root / "reference.jsonl",
                                     reference_exit_code=0, output=cls.root / "output", mode="batch",
                                     worker_config=cls.root / "configuration.json")
        cls.options.tasks.write_text(json.dumps(replay.fixture.declarations()))
        replay.fixture.write(cls.options.reference_log, reference())
        _, _, calls = direct.calls(cls.options, mode="batch")
        cls.count = len(calls)
        cls.options.worker.write_text(batch.source(calls, adapter=cls.options.adapter, configuration=cls.options.worker_config,
                                                   load_seconds=fixture.LOAD_SECONDS, inference_seconds=fixture.INFERENCE_SECONDS))
        serial = replace(cls.options, mode="session", worker=cls.root / "serial.py")
        serial.worker.write_text(replay.session_source(calls, load_seconds=fixture.LOAD_SECONDS, inference_seconds=fixture.INFERENCE_SECONDS))
        services = direct.Services(run=subprocess.run, clock=time.perf_counter)
        cls.runs, cls.serial = [], []
        for index, elapsed in enumerate(fixture.INVAR_DURATIONS):
            path = cls.root / f"invar-{index}.jsonl"
            replay.fixture.write(path, reference())
            cls.runs.append({"name": f"invar-{index}", "route": "invar", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})
        for index, elapsed in enumerate(fixture.DIRECT_DURATIONS):
            for options, runs in ((cls.options, cls.runs), (serial, cls.serial)):
                path = cls.root / f"{options.mode}-{index}"
                direct.run(replace(options, output=path), services)
                runs.append({"name": f"{options.mode}-{index}", "route": "direct", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})

    def report(self, runs):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"reference_log": str(self.options.reference_log), "reference_exit_code": 0, "runs": runs}))
        return performance.report(path, self.options.tasks, self.options.policy)

    def test_finite_execution_duration_is_counted_once_with_all_logical_tokens(self):
        report = self.report(self.runs)
        self.assertTrue(report["comparison"]["all_results_equal_to_reference"])
        measured = evidence.measurements(self.options.reference_log, self.options.tasks, self.options.policy, exit_code=0)
        self.assertEqual([row["requests_per_execution"] for row in measured["loads"]], [[2], [4]])
        self.assertEqual(len(measured["measurements"]), 2)
        self.assertEqual([len(row["calls"]) for row in measured["measurements"]], [2, 4])
        for row in report["runs"]:
            self.assertAlmostEqual(row["inference_seconds"]["total"], 2 * fixture.INFERENCE_SECONDS)
            self.assertAlmostEqual(row["worker_critical_path_seconds"], 2 * (fixture.LOAD_SECONDS + fixture.INFERENCE_SECONDS))
            self.assertEqual(row["response_tokens"], self.count * len(replay.RESPONSE))

    def test_equal_load_counts_do_not_allow_serial_execution_to_replace_batches(self):
        with self.assertRaisesRegex(ValueError, "execution"):
            self.report([*self.runs[:2], *self.serial])

    def test_raw_inventory_and_duration_cannot_be_fabricated_in_summary(self):
        rows = reference()
        consumed = next(index for index, row in enumerate(rows) if row.get("stage") == "consumed")
        result = consumed + 2
        changed = copy.deepcopy(rows)
        changed[result]["calls"].pop()
        variants = (changed, rows[:consumed + 1] + rows[consumed + 2:],
                    rows[:consumed + 1] + [rows[consumed + 1]] + rows[consumed + 1:])
        path = self.root / "invalid.jsonl"
        for invalid in variants:
            replay.fixture.write(path, invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.report([{**self.runs[0], "path": str(path)}, *self.runs[1:]])
        completion = Path(self.runs[-1]["path"]) / "complete.json"
        original = completion.read_bytes()
        try:
            for change in ({"loads": self.count}, {"mode": "session"}):
                completion.write_text(json.dumps({**json.loads(original), **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.report(self.runs)
        finally:
            completion.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
