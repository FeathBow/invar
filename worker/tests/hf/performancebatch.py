import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import json
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from worker import direct
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
        services = direct.Services(run=subprocess.run, clock=time.perf_counter)
        cls.runs = []
        for index, elapsed in enumerate(fixture.INVAR_DURATIONS):
            path = cls.root / f"invar-{index}.jsonl"
            replay.fixture.write(path, reference())
            cls.runs.append({"name": f"invar-{index}", "route": "invar", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})
        for index, elapsed in enumerate(fixture.DIRECT_DURATIONS):
            path = cls.root / f"batch-{index}"
            direct.run(replace(cls.options, output=path), services)
            cls.runs.append({"name": f"batch-{index}", "route": "direct", "path": str(path), "exit_code": 0, "elapsed_seconds": elapsed})

    def report(self, runs):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"reference_log": str(self.options.reference_log), "reference_exit_code": 0, "runs": runs}))
        return fixture.summarize(path, self.options.tasks, self.options.policy)

    def test_finite_execution_duration_is_counted_once_with_all_logical_tokens(self):
        report = self.report(self.runs)
        self.assertTrue(report["comparison"]["all_results_equal_to_reference"])
        for row in report["runs"]:
            self.assertAlmostEqual(row["inference_seconds"]["total"], 2 * fixture.INFERENCE_SECONDS)
            self.assertAlmostEqual(row["worker_critical_path_seconds"], 2 * (fixture.LOAD_SECONDS + fixture.INFERENCE_SECONDS))
            self.assertEqual(row["response_tokens"], self.count * len(replay.RESPONSE))

    def test_changed_batch_completion_is_rejected(self):
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
