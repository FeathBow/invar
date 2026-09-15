import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path

from worker import direct
from worker import evaluation
from worker.tests import quality as fixture

FAILURE_STATUS = 19
PREFIX = (11, 13)
RESPONSE = (17, 23)
BEHAVIOR_WORD = 3212836864
TOKENIZER = "c" * 64
BASE = "e" * 64
ASSEMBLY = "f" * 64


def call(task, binding):
    consumed = {"stage": "consumed", "binding": binding, "program": "protocol test fixture",
                "adapter": fixture.INITIAL_POLICY,
                "request": {key: task[key] for key in ("prompt", "seed", "tokens", "temperature")}}
    result = {"stage": "result", "binding": binding, "adapter": fixture.INITIAL_POLICY,
              "request": consumed["request"], "tokens": [*PREFIX, *RESPONSE], "prompt_length": len(PREFIX),
              "behavior": [-1.0] * len(RESPONSE), "behavior_bits": [BEHAVIOR_WORD] * len(RESPONSE),
              "text": "protocol fixture", "truncated": False}
    return direct.Call(consumed=consumed, result=result)


def reference():
    summaries = fixture.records(fixture.INITIAL_POLICY, ((0, 1), (1, 0, 1, 0)))
    tasks = fixture.declarations()
    result = []
    for index, summary in enumerate(summaries[:-1]):
        for task, sample in zip(tasks[index]["tasks"], summary["samples"], strict=True):
            bound = call(task, sample["binding"])
            result.extend((bound.consumed, bound.result))
        result.append(summary)
    return [*result, summaries[-1]]


def output(bound):
    def measure(stage):
        return {"stage": stage, "seconds": 1.0, "peak_allocated": 100, "peak_reserved": 200}
    return [measure("load"), bound.consumed, measure("inference"), bound.result]


def session_reference():
    rows = materialized_reference()
    return [{**row, "load": {"binding": row["binding"], "program": "explicit load reference fixture"}}
            if row.get("stage") == "consumed" else row for row in rows]


def session_source(calls, *, load_seconds=1.0, inference_seconds=1.0):
    payload = {bound.consumed["binding"]["call"]: bound.result for bound in calls}
    return "\n".join((
        "import json,sys",
        "lines=[json.loads(line) for line in sys.stdin.read().splitlines()]",
        "assert lines and len(lines)%2==0",
        "pairs=list(zip(lines[0::2],lines[1::2]))",
        "assert all(second=={key:first[key] for key in ('binding','program')} for first,second in pairs)",
        f"payload={payload!r}",
        "def emit(row): print(json.dumps(row),flush=True)",
        "emit({'stage':'loading'})",
        "emit({'stage':'profile','precision':'fixture'})",
        f"emit({{'stage':'load','seconds':{load_seconds!r},'peak_allocated':100,'peak_reserved':200}})",
        "for index,(first,_) in enumerate(pairs):",
        "    if index: emit({'stage':'unloaded_adapter',**pairs[index-1][0]['load']})",
        "    emit({'stage':'loaded_adapter','binding':first['binding'],'requested':first['adapter'],'consumed':first['adapter'],'model':'protocol-fixture','revision':'fixed-fixture','scope':'fixture only',**{key:first[key] for key in ('tokenizer','base','assembly')}})",
        "    emit({'stage':'consumed',**first})",
        f"    emit({{'stage':'inference','seconds':{inference_seconds!r},'peak_allocated':100,'peak_reserved':200}})",
        "    emit(payload[first['binding']['call']])",
        "",
    ))


def tokenizer_reference():
    return [{**row, **({"tokenizer": TOKENIZER} if row.get("stage") in ("consumed", "result")
                      or row.get("phase") == "evaluation_complete" else {})} for row in reference()]


def materialized_reference():
    return [{**row, **({"base": BASE, "assembly": ASSEMBLY} if "tokenizer" in row else {})}
            for row in tokenizer_reference()]


class DirectChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="invar-direct-test-"))
        self.options = direct.Options(python=sys.executable, worker=self.root / "worker.py", cache=self.root,
                                      adapter=self.root / "adapter", policy=fixture.INITIAL_POLICY,
                                      tasks=self.root / "tasks.json", reference_log=self.root / "reference.jsonl",
                                      reference_exit_code=0, output=self.root / "output", mode="process")
        self.options.tasks.write_text(json.dumps(fixture.declarations()))
        fixture.write(self.options.reference_log, reference())
        self.services = direct.Services(run=subprocess.run, clock=time.perf_counter)

    def test_complete_inventory_and_exact_arguments(self):
        digest, tasks_digest, calls = direct.calls(self.options)
        self.assertEqual(digest, evaluation.snapshot(self.options.reference_log)[0])
        self.assertEqual(tasks_digest, evaluation.snapshot(self.options.tasks)[0])
        self.assertEqual(len(calls), sum(len(row["tasks"]) for row in fixture.declarations()))
        self.assertEqual([value.consumed["binding"]["call"] for value in calls], list(range(len(calls))))
        for bound in calls:
            args = direct.command(bound, self.options)
            self.assertIn(f"--seed={bound.consumed['request']['seed']}", args)
            self.assertIn(f"--digest={fixture.INITIAL_POLICY}", args)

    def test_selected_core_owns_reference_and_output_admission(self):
        executable = shutil.which("invar")
        self.assertIsNotNone(executable)
        _, _, planned = direct.calls(self.options, core_executable=executable)
        path = self.root / "selected-core.jsonl"
        fixture.write(path, output(planned[0]))
        self.assertTrue(direct.inspect(path, planned[0], exit_code=0, core_executable=executable)["result_equal"])
        with self.assertRaisesRegex(ValueError, "did not exit successfully"):
            direct.inspect(path, planned[0], exit_code=FAILURE_STATUS, core_executable=executable)
        with self.assertRaises(FileNotFoundError):
            direct.calls(self.options, core_executable=self.root / "missing-core")
        with self.assertRaises(FileNotFoundError):
            direct.inspect(path, planned[0], exit_code=0, core_executable=self.root / "missing-core")

    def test_core_normalizes_integral_spellings_without_losing_probability_zero_sign(self):
        rows = reference()
        for row in rows:
            if "binding" in row:
                row["binding"] = {key: float(value) for key, value in row["binding"].items()}
            if "request" in row:
                row["request"] = {**row["request"], **{key: float(row["request"][key]) for key in ("seed", "tokens")}}
            if row.get("stage") == "result":
                row["tokens"] = list(map(float, row["tokens"]))
                row["prompt_length"] = float(row["prompt_length"])
                row["behavior_bits"] = [float(0x80000000), *map(float, row["behavior_bits"][1:])]
                row["behavior"][0] = -0.0
        fixture.write(self.options.reference_log, rows)
        _, _, planned = direct.calls(self.options)
        for observed in planned:
            self.assertTrue(all(type(value) is int for value in observed.consumed["binding"].values()))
            self.assertTrue(all(type(observed.consumed["request"][key]) is int for key in ("seed", "tokens")))
            self.assertTrue(all(type(value) is int for value in observed.result["tokens"] + observed.result["behavior_bits"]))
            self.assertEqual(struct.pack("!d", observed.result["behavior"][0]), struct.pack("!d", -0.0))

    def test_core_preserves_explicit_cpu_timing_and_rejects_ambiguous_or_reordered_measurements(self):
        _, _, planned = direct.calls(self.options)
        rows = output(planned[0])
        rows[0] = {"stage": "load", "cpu_seconds": 0.1}
        rows[2] = {"stage": "inference", "cpu_seconds": 0.2}
        path = self.root / "cpu-timing.jsonl"
        fixture.write(path, rows)
        result = direct.inspect(path, planned[0], exit_code=0)
        self.assertEqual(result["load"], {"cpu_seconds": 0.1})
        self.assertEqual(result["inference"], {"cpu_seconds": 0.2})
        for changed in ([rows[2], rows[1], rows[0], rows[3]],
                        [{**rows[0], "seconds": 0.1}, *rows[1:]],
                        [rows[0], rows[1], {**rows[2], "cpu_seconds": -1}, rows[3]]):
            fixture.write(path, changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                direct.inspect(path, planned[0], exit_code=0)

    def test_reference_policy_and_consumption_order_are_checked_before_launch(self):
        rows = reference()
        rows[0]["adapter"] = rows[1]["adapter"] = fixture.TRAINED_POLICY
        fixture.write(self.options.reference_log, rows)
        with self.assertRaisesRegex(ValueError, "Reference policy mismatch"):
            direct.run(self.options, self.services)
        rows = reference()
        rows[0], rows[1] = rows[1], rows[0]
        fixture.write(self.options.reference_log, rows)
        with self.assertRaisesRegex(ValueError, "precedes its consumption"):
            direct.run(self.options, self.services)
        self.assertFalse(self.options.output.exists())

    def test_load_program_is_replayed_separately_from_computation_permission(self):
        rows = materialized_reference()
        for row in rows:
            if row.get('stage') == 'consumed':
                row['load'] = {'binding': row['binding'], 'program': 'explicit load reference fixture'}
        fixture.write(self.options.reference_log, rows)
        _, _, calls = direct.calls(self.options)
        for bound in calls:
            initial = json.loads(direct.envelope(bound))
            permitted = json.loads(direct.permission(bound))
            self.assertEqual(initial['load'], bound.consumed['load'])
            self.assertEqual(set(permitted), {'binding', 'program'})
            self.assertEqual(permitted, {key: initial[key] for key in ('binding', 'program')})
        rows[0]['load']['binding'] = {**rows[0]['binding'], 'instance': 999}
        fixture.write(self.options.reference_log, rows)
        with self.assertRaisesRegex(ValueError, 'load and inference bindings differ'):
            direct.calls(self.options)

    def test_tokenizer_identity_is_preserved_in_arguments_and_results(self):
        fixture.write(self.options.reference_log, tokenizer_reference())
        _, _, calls = direct.calls(self.options)
        self.assertIn(f"--tokenizer-digest={TOKENIZER}", direct.command(calls[0], self.options))
        records = output(calls[0])
        path = self.root / "tokenizer.jsonl"
        fixture.write(path, records)
        self.assertTrue(direct.inspect(path, calls[0], exit_code=0)["result_equal"])
        fixture.write(path, [*records[:-1], {**records[-1], "tokenizer": "d" * 64}])
        with self.assertRaisesRegex(ValueError, "tokenizer binding"):
            direct.inspect(path, calls[0], exit_code=0)

    def test_missing_or_mixed_tokenizer_bindings_cannot_form_a_reference(self):
        for index, value in product((0, 1, -1), (None, "d" * 64)):
            rows = tokenizer_reference()
            rows[index] = {key: item for key, item in rows[index].items() if key != "tokenizer"}
            if value is not None:
                rows[index] = {**rows[index], "tokenizer": value}
            fixture.write(self.options.reference_log, rows)
            with self.subTest(index=index, value=value), self.assertRaisesRegex(ValueError, "tokenizer"):
                direct.calls(self.options)

    def test_materialization_is_preserved_in_replay_and_result_checks(self):
        fixture.write(self.options.reference_log, materialized_reference())
        _, _, calls = direct.calls(self.options)
        args = direct.command(calls[0], self.options)
        self.assertIn(f"--base-digest={BASE}", args)
        self.assertIn(f"--assembly-digest={ASSEMBLY}", args)
        path = self.root / "materialized.jsonl"
        records = output(calls[0])
        fixture.write(path, records)
        self.assertTrue(direct.inspect(path, calls[0], exit_code=0)["result_equal"])
        for name in ("base", "assembly"):
            fixture.write(path, [*records[:-1], {**records[-1], name: "0" * 64}])
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "binding differs"):
                direct.inspect(path, calls[0], exit_code=0)

    def test_incomplete_or_changed_materialization_cannot_form_a_reference(self):
        for index, name in product((0, 1, -1), ("base", "assembly")):
            rows = materialized_reference()
            rows[index] = {key: value for key, value in rows[index].items() if key != name}
            fixture.write(self.options.reference_log, rows)
            with self.subTest(index=index, field=name), self.assertRaisesRegex(ValueError, "materialization"):
                direct.calls(self.options)
        rows = materialized_reference()
        rows[0] = {**rows[0], "assembly": "0" * 64}
        fixture.write(self.options.reference_log, rows)
        with self.assertRaisesRegex(ValueError, "bindings disagree"):
            direct.calls(self.options)

    def test_invalid_reference_never_launches(self):
        changes = (
            lambda rows: rows[:-1],
            lambda rows: [rows[0], *rows],
            lambda rows: [{**rows[0], "request": {**rows[0]["request"], "seed": -1}}, *rows[1:]],
        )
        for change in changes:
            with self.subTest(change=change):
                fixture.write(self.options.reference_log, change(reference()))
                with self.assertRaises(ValueError):
                    direct.run(self.options, self.services)
                self.assertFalse(self.options.output.exists())

    def script(self, bound, *, failure=False):
        records = output(bound)
        source = "import json,sys\nfirst=json.loads(sys.stdin.readline())\nsecond=json.loads(sys.stdin.readline())\n"
        source += f"expected={json.loads(direct.envelope(bound))!r}\nassert first==second==expected\n"
        source += "sys.stderr.write('retained diagnostic\\n')\n"
        source += f"sys.exit({FAILURE_STATUS})\n" if failure else f"records={records!r}\nfor row in records: print(json.dumps(row),flush=True)\n"
        self.options.worker.write_text(source)

    def test_real_process_preserves_handshake_and_measures_output(self):
        _, _, calls = direct.calls(self.options)
        bound = calls[0]
        self.script(bound)
        self.options.output.mkdir()
        result = direct.execute(bound, self.options, services=self.services, index=0)
        self.assertTrue(result["result_equal"])
        self.assertEqual(result["response_tokens"], len(RESPONSE))
        self.assertGreater(result["process_seconds"], 0)
        self.assertEqual((self.options.output / "0000.stderr.log").read_text(), "retained diagnostic\n")
        self.assertEqual(json.loads((self.options.output / "0000.status.json").read_text())["exit_code"], 0)

    def test_nonzero_process_retains_failure_and_no_completion(self):
        _, _, calls = direct.calls(self.options)
        self.script(calls[0], failure=True)
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            direct.run(self.options, self.services)
        self.assertEqual(failure.exception.returncode, FAILURE_STATUS)
        status = json.loads((self.options.output / "0000.status.json").read_text())
        self.assertEqual(status["exit_code"], FAILURE_STATUS)
        self.assertFalse((self.options.output / "complete.json").exists())
        self.assertEqual((self.options.output / "calls.jsonl").read_bytes(), b"")

    def test_full_real_process_inventory_has_one_completion(self):
        _, _, calls = direct.calls(self.options)
        payload = {bound.consumed["binding"]["call"]: output(bound) for bound in calls}
        source = "import json,sys\nfirst=json.loads(sys.stdin.readline())\nsecond=json.loads(sys.stdin.readline())\n"
        source += "assert first==second\n"
        source += f"payload={payload!r}\nrecords=payload[first['binding']['call']]\n"
        source += "assert '--seed='+str(records[1]['request']['seed']) in sys.argv\n"
        source += "for row in records: print(json.dumps(row),flush=True)\n"
        self.options.worker.write_text(source)
        report = direct.run(self.options, self.services)
        self.assertEqual(report["calls"], len(calls))
        self.assertEqual(report["equal_results"], len(calls))
        self.assertEqual(report["response_tokens"], len(calls) * len(RESPONSE))
        self.assertEqual(report, json.loads((self.options.output / "complete.json").read_text()))
        self.assertEqual(len((self.options.output / "calls.jsonl").read_text().splitlines()), len(calls))

    def test_numerical_difference_is_reported(self):
        _, _, calls = direct.calls(self.options)
        records = copy.deepcopy(output(calls[0]))
        records[-1]["behavior_bits"][0] += 1
        word = records[-1]["behavior_bits"][0]
        records[-1]["behavior"][0] = struct.unpack("!f", struct.pack("!I", word))[0]
        path = self.root / "different.jsonl"
        fixture.write(path, records)
        self.assertFalse(direct.inspect(path, calls[0], exit_code=0)["result_equal"])

    def test_inconsistent_behavior_is_rejected_on_both_input_paths(self):
        _, _, calls = direct.calls(self.options)
        variants = ((-1.0, BEHAVIOR_WORD + 1), (-1.0, 0x7FC00000), (-1.0, 0xFF800000),
                    (-1.0, 0x3F800000), (0.0, 0x80000000), (-0.0, 0), (-0.1, 0xBDCCCCCD))
        for value, word in variants:
            with self.subTest(value=value, word=word):
                rows = reference()
                rows[1]["behavior"][0] = value
                rows[1]["behavior_bits"][0] = word
                fixture.write(self.options.reference_log, rows)
                with self.assertRaises(ValueError):
                    direct.calls(self.options)
                records = copy.deepcopy(output(calls[0]))
                records[-1]["behavior"][0] = value
                records[-1]["behavior_bits"][0] = word
                path = self.root / "inconsistent.jsonl"
                fixture.write(path, records)
                with self.assertRaises(ValueError):
                    direct.inspect(path, calls[0], exit_code=0)

    def test_exact_fp32_boundaries_remain_admissible(self):
        _, _, calls = direct.calls(self.options)
        for word in (0, 0x80000000, 0x80000001, 0xFF7FFFFF):
            with self.subTest(word=word):
                records = copy.deepcopy(output(calls[0]))
                records[-1]["behavior_bits"][0] = word
                records[-1]["behavior"][0] = struct.unpack("!f", struct.pack("!I", word))[0]
                path = self.root / "boundary.jsonl"
                fixture.write(path, records)
                self.assertEqual(direct.inspect(path, calls[0], exit_code=0)["response_tokens"], len(RESPONSE))

    def test_incomplete_and_misbound_output_rejected(self):
        _, _, calls = direct.calls(self.options)
        rows = output(calls[0])
        changes = (
            lambda: rows[:-1],
            lambda: [*rows, rows[-1]],
            lambda: [*rows[:-1], {**rows[-1], "binding": dict.fromkeys(("call", "attempt", "instance"), 999)}],
            lambda: [*rows[:-1], {**rows[-1], "behavior_bits": []}],
        )
        for change in changes:
            with self.subTest(change=change):
                path = self.root / "invalid.jsonl"
                fixture.write(path, change())
                with self.assertRaises(ValueError):
                    direct.inspect(path, calls[0], exit_code=0)

    def test_session_replay_queues_every_request_behind_one_load(self):
        fixture.write(self.options.reference_log, session_reference())
        options = replace(self.options, mode="session")
        _, _, calls = direct.calls(options)
        self.options.worker.write_text(session_source(calls))
        report = direct.run(options, self.services)
        grouped = direct.cohorts(calls)
        self.assertEqual((report["mode"], report["calls"], report["cohorts"], report["loads"], report["equal_results"]),
                         ("session", len(calls), len(grouped), len(grouped), len(calls)))
        self.assertEqual(report["response_tokens"], len(calls) * len(RESPONSE))
        self.assertEqual(report, json.loads((options.output / "complete.json").read_text()))
        rows = [json.loads(line) for line in (options.output / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([(row["cohort"], row["index"]) for row in rows],
                         [(index, position) for index, members in grouped for position in range(len(members))])
        self.assertTrue(all(row["result_equal"] for row in rows))
        for index, members in grouped:
            status = json.loads((options.output / f"session-{index:04d}.status.json").read_text())
            self.assertEqual((status["cohort"], status["exit_code"], status["calls"]), (index, 0, len(members)))
        queued = direct.session_input(calls).splitlines()
        self.assertEqual(len(queued), 2 * len(calls))
        self.assertEqual(json.loads(queued[1]), {key: calls[0].consumed[key] for key in ("binding", "program")})
        self.assertEqual(set(json.loads(queued[0])), set(direct.SESSION_FIELDS.split()))

    def test_session_replay_requires_load_and_materialization_before_launching(self):
        for rows in (materialized_reference(), reference()):
            fixture.write(self.options.reference_log, rows)
            options = replace(self.options, mode="session")
            with self.subTest(rows=len(rows)), self.assertRaisesRegex(ValueError, "batch reference"):
                direct.run(options, self.services)
            self.assertFalse(options.output.exists())

    def test_session_output_with_repeated_loads_or_reordered_stages_is_rejected(self):
        fixture.write(self.options.reference_log, session_reference())
        options = replace(self.options, mode="session")
        _, _, calls = direct.calls(options)
        self.options.worker.write_text(session_source(calls))
        options.output.mkdir()
        observed = direct.execute_session(calls, options, services=self.services, queued=direct.session_input(calls), index=0)
        self.assertEqual(len(observed["calls"]), len(calls))
        rows = [json.loads(line) for line in (options.output / "session-0000.stdout.jsonl").read_text().splitlines()]
        load = next(row for row in rows if row["stage"] == "load")
        results = [index for index, row in enumerate(rows) if row["stage"] == "result"]
        swapped = list(rows)
        swapped[results[0]], swapped[results[1]] = rows[results[1]], rows[results[0]]
        changes = (lambda: [*rows, load], lambda: swapped, lambda: rows[:-1], lambda: [*rows, rows[-1]],
                   lambda: [row for row in rows if row["stage"] != "load"])
        for change in changes:
            target = self.root / "session-invalid.jsonl"
            fixture.write(target, change())
            with self.subTest(change=change), self.assertRaises(ValueError):
                direct.inspect_session(target, calls, exit_code=0)

    def test_existing_output_is_preserved(self):
        self.options.output.mkdir()
        marker = self.options.output / "existing"
        marker.write_bytes(b"unchanged")
        with self.assertRaises(FileExistsError):
            direct.run(self.options, self.services)
        self.assertEqual(marker.read_bytes(), b"unchanged")

    def test_loaded_and_unloaded_records_must_match_actual_consumption(self):
        fixture.write(self.options.reference_log, session_reference())
        options = replace(self.options, mode="session")
        _, _, calls = direct.calls(options)
        self.options.worker.write_text(session_source(calls))
        options.output.mkdir()
        direct.execute_session(calls, options, services=self.services, queued=direct.session_input(calls), index=0)
        rows = [json.loads(line) for line in (options.output / "session-0000.stdout.jsonl").read_text().splitlines()]
        for stage, key, value in (("loaded_adapter", "requested", "0" * 64), ("loaded_adapter", "model", ""),
                                  ("unloaded_adapter", "program", "wrong prior load")):
            changed = copy.deepcopy(rows)
            next(row for row in changed if row["stage"] == stage)[key] = value
            target = self.root / "load-invalid.jsonl"
            fixture.write(target, changed)
            with self.subTest(stage=stage, key=key), self.assertRaises(ValueError):
                direct.inspect_session(target, calls, exit_code=0)


if __name__ == "__main__":
    unittest.main()
