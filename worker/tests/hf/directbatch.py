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
from worker.vllm import entry as vllm_entry
from worker.batch import FORMAT
from worker.tests.hf import direct as fixture


def encoded(rows):
    return "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)


def output(calls, *, load_seconds=1.0, inference_seconds=2.0):
    def loaded(call):
        fields = call.consumed
        return {"stage": "loaded_adapter", "binding": fields["binding"],
                "requested": fields["adapter"], "consumed": fields["adapter"],
                "model": "protocol-fixture", "revision": "fixed-fixture", "scope": "fixture only",
                **{key: fields[key] for key in ("tokenizer", "base", "assembly")}}

    def measure(stage, seconds):
        return {"stage": stage, "seconds": seconds, "peak_allocated": 100, "peak_reserved": 200}

    return [{"stage": "profile", "precision": "fixture"}, measure("load", load_seconds),
            {"stage": "consumed", "format": FORMAT,
             "calls": [encoded((loaded(call), call.consumed)) for call in calls]},
            measure("inference", inference_seconds),
            {"stage": "result", "format": FORMAT, "calls": [encoded((call.result,)) for call in calls]}]


def source(calls, *, adapter, configuration, load_seconds=1.0, inference_seconds=2.0):
    groups = direct.cohorts(calls)
    payload = {tuple(call.consumed["binding"]["call"] for call in members):
               output(members, load_seconds=load_seconds, inference_seconds=inference_seconds)
               for _, members in groups}
    return "\n".join((
        "import json,sys",
        "frames=[json.loads(line) for line in sys.stdin.read().splitlines()]",
        "assert len(frames)==2",
        "first,second=frames",
        f"assert first['format']==second['format']=={FORMAT!r}",
        f"assert first['adapter']=={str(adapter)!r}",
        "assert not any(arg.startswith('--adapter=') for arg in sys.argv)",
        f"assert {'--config=' + str(configuration)!r} in sys.argv",
        "calls=[json.loads(raw) for raw in first['calls']]",
        "permissions=[json.loads(raw) for raw in second['permissions']]",
        "assert permissions==[{key:call[key] for key in ('binding','program')} for call in calls]",
        f"payload={payload!r}",
        "key=tuple(call['binding']['call'] for call in calls)",
        "for row in payload[key]: print(json.dumps(row),flush=True)",
        "",
    ))


class DirectBatchChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="invar-direct-batch-"))
        self.options = direct.Options(python=sys.executable, worker=self.root / "worker.py", cache=self.root,
                                      adapter=self.root / "adapter", policy=fixture.fixture.INITIAL_POLICY,
                                      tasks=self.root / "tasks.json", reference_log=self.root / "reference.jsonl",
                                      reference_exit_code=0, output=self.root / "output", mode="batch",
                                      worker_config=self.root / "native-config.json")
        self.options.tasks.write_text(json.dumps(fixture.fixture.declarations()))
        fixture.fixture.write(self.options.reference_log, fixture.session_reference())
        self.services = direct.Services(run=subprocess.run, clock=time.perf_counter)

    def test_complete_process_replay_records_one_duration_for_each_finite_group(self):
        _, _, calls = direct.calls(self.options, mode="batch")
        self.options.worker.write_text(source(calls, adapter=self.options.adapter, configuration=self.options.worker_config))
        report = direct.run(self.options, self.services)
        groups = direct.cohorts(calls)
        self.assertEqual((report["mode"], report["loads"], report["equal_results"]), ("batch", len(groups), len(calls)))
        rows = [json.loads(line) for line in (self.options.output / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), len(calls))
        self.assertTrue(all("inference" not in row and row["result_equal"] for row in rows))
        for index, members in groups:
            path = self.options.output / f"session-{index:04d}.stdout.jsonl"
            observed = direct.inspect_session(path, members, mode="batch", exit_code=0)
            self.assertEqual(observed["inference"]["seconds"], 2.0)
            self.assertEqual(len(observed["calls"]), len(members))

    def test_group_is_exact_and_configuration_reaches_every_explicit_mode(self):
        _, _, calls = direct.calls(self.options, mode="batch")
        frames = [json.loads(line) for line in direct.batch_input(calls, adapter=self.options.adapter).splitlines()]
        self.assertEqual(frames[0]["calls"], [direct.session_envelope(call).decode() for call in calls])
        self.assertEqual(frames[1]["permissions"], [direct.permission(call).decode() for call in calls])
        for mode in direct.MODES:
            options = replace(self.options, mode=mode)
            args = direct.command(calls[0], options) if mode == "process" else direct.session_command(options)
            configured, _ = vllm_entry.parser("Native worker configuration").parse_known_args(args[2:])
            self.assertEqual(configured.config, options.worker_config)
            self.assertEqual(any(arg.startswith("--adapter=") for arg in args), mode not in ("batch", "resident"))

    def test_incomplete_materialization_fails_before_launch(self):
        for rows in (fixture.reference(), fixture.materialized_reference()):
            fixture.fixture.write(self.options.reference_log, rows)
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                direct.run(self.options, self.services)
            self.assertFalse(self.options.output.exists())

    def test_actual_output_cannot_substitute_serial_or_incomplete_groups(self):
        _, _, calls = direct.calls(self.options, mode="batch")
        rows = output(calls)
        reordered = copy.deepcopy(rows)
        reordered[-1]["calls"].reverse()
        incomplete = copy.deepcopy(rows)
        incomplete[-1]["calls"].pop()
        variants = (reordered, incomplete, rows[:-1], [*rows, rows[-1]], [*rows[:3], rows[3], *rows[3:]])
        path = self.root / "invalid.jsonl"
        for changed in variants:
            fixture.fixture.write(path, changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                direct.inspect_session(path, calls, mode="batch", exit_code=0)
        fixture.fixture.write(path, rows)
        with self.assertRaises(ValueError):
            direct.inspect_session(path, calls, mode="session", exit_code=0)

    def test_failed_child_preserves_status_without_completion(self):
        self.options.worker.write_text(f"import sys\nsys.stdin.read()\nsys.exit({fixture.FAILURE_STATUS})\n")
        with self.assertRaises(subprocess.CalledProcessError):
            direct.run(self.options, self.services)
        status = json.loads((self.options.output / "session-0000.status.json").read_text())
        self.assertEqual(status["exit_code"], fixture.FAILURE_STATUS)
        self.assertFalse((self.options.output / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
