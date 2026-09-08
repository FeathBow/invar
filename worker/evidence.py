from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import direct
import evaluation
from cohort import fields, identity, number

BLOCK = ("loaded_adapter", "consumed", "inference", "result")


@dataclass(frozen=True, kw_only=True)
class Reference:
    tasks: Path
    policy: str
    reference_log: Path
    reference_exit_code: int


def records(path):
    digest, encoded = evaluation.snapshot(path)
    evaluation.require(encoded.endswith(b"\n"), "Incomplete measurement record stream")
    return digest, tuple(evaluation.decode(line) for line in encoded.splitlines())


def document(path):
    digest, encoded = evaluation.snapshot(path)
    return digest, evaluation.decode(encoded)


def positive(value):
    result = number(value)
    evaluation.require(result > 0, "Expected a positive measurement duration")
    return result


def sessions(rows):
    grouped, current, block = [], None, []
    for row in rows:
        if "stage" not in row:
            continue
        stage = row["stage"]
        if stage == "loading":
            evaluation.require(not block, "Model loading reported inside a result block")
        elif stage == "profile":
            evaluation.require(not block, "Numerical profile reported inside a result block")
            current = {"profile": row, "load": None, "blocks": []}
            grouped.append(current)
        elif stage == "load":
            evaluation.require(current is not None and current["load"] is None and not current["blocks"] and not block,
                               "Model load without its own preceding profile")
            current["load"] = direct.measured(row)
        elif stage == "unloaded_adapter":
            evaluation.require(not block and current is not None and bool(current["blocks"]),
                               "Adapter replacement outside a completed result block")
        else:
            evaluation.require(current is not None and current["load"] is not None,
                               "Worker result block outside a loaded model session")
            block.append(row)
            if stage == "result":
                current["blocks"].append(tuple(block))
                block = []
    evaluation.require(not block, "Incomplete worker measurement block")
    evaluation.require(bool(grouped) and all(session["load"] is not None and session["blocks"] for session in grouped),
                       "Model session without a load or without results")
    return grouped


def measurement(session, block):
    evaluation.require(tuple(row["stage"] for row in block) == BLOCK, "Missing, repeated or reordered worker stages in a result block")
    loaded, consumed, timing, result = block
    evaluation.require(loaded["binding"] == consumed["binding"] == result["binding"], "Result block bindings disagree")
    return {"inference": direct.measured(timing), "response_tokens": direct.result_tokens(result),
            "profile_sha256": profile(session["profile"], block)}


def measured_stream(rows):
    grouped = sessions(rows)
    measured = tuple(measurement(session, block) for session in grouped for block in session["blocks"])
    loads = []
    for session in grouped:
        inference = sum(measurement(session, block)["inference"]["seconds"] for block in session["blocks"])
        loads.append({**session["load"], "inference_seconds": inference, "calls": len(session["blocks"])})
    return measured, tuple(loads)


def profile(reported, block):
    loaded = tuple(row for row in block if row["stage"] == "loaded_adapter")
    evaluation.require(len(loaded) == 1 and reported.get("stage") == "profile", "Missing or repeated numerical profile or model identity")
    value = loaded[0]
    bindings = evaluation.model_binding(value)
    fields(value, "stage binding requested consumed model revision scope " + " ".join(bindings))
    evaluation.require(all(isinstance(value[key], str) and value[key] for key in ("model", "revision")), "Missing model identity")
    identity(value["requested"])
    identity(value["consumed"])
    evaluation.require(value["requested"] == value["consumed"] == block[-1]["adapter"]
                       and value["binding"] == block[-1]["binding"], "Loaded model binding differs from the result")
    evaluation.require(bindings == evaluation.model_binding(block[-1]), "Loaded materialization differs from the result")
    encoded = json.dumps({"profile": reported, "model": value["model"], "revision": value["revision"], **bindings},
                         sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def summaries(rows):
    samples = {evaluation.binding(sample["binding"]): sample
               for row in rows if row.get("phase") == "evaluation" for sample in row["samples"]}
    for row in rows:
        if row.get("stage") == "result":
            expected = samples[evaluation.binding(row["binding"])]
            evaluation.require(direct.result_tokens(row) == expected["response_tokens"]
                               and row["truncated"] == expected["truncated"], "Worker result differs from evaluation summary")


def cohort_streams(rows):
    chunks, current = [], []
    for row in rows:
        if row.get("phase") == "evaluation":
            chunks.append(tuple(current))
            current = []
        elif "stage" in row:
            current.append(row)
    evaluation.require(not current, "Worker records after the final evaluation cohort")
    return tuple(chunks)


def schedule(loads, concurrent):
    per_session = tuple(row["seconds"] + row["inference_seconds"] for row in loads)
    return max(per_session) if concurrent else sum(per_session)


def invar(run, reference, expected):
    source = Reference(tasks=reference.tasks, policy=reference.policy,
                       reference_log=Path(run["path"]), reference_exit_code=run["exit_code"])
    digest, _, calls = direct.calls(source)
    observed_digest, rows = records(source.reference_log)
    evaluation.require(digest == observed_digest, "Evaluation log changed during measurement")
    evaluation.require(len(calls) == len(expected), "Mismatched repeated evaluation inventory")
    for call, first in zip(calls, expected, strict=True):
        evaluation.require(call.consumed == first.consumed, "Repeated evaluation consumed different inputs or order")
    summaries(rows)
    declared = rows[-1].get("sessions")
    declared = None if declared is None else evaluation.natural(declared)
    concurrent = declared is not None and declared > 1
    measured, loads, counts, critical = [], [], [], 0.0
    for index, chunk in enumerate(cohort_streams(rows)):
        blocks, cohort_loads = measured_stream(chunk)
        if declared is not None:
            evaluation.require(len(cohort_loads) == declared, "Declared session count differs from the recorded model loads of a cohort")
        measured.extend(blocks)
        loads.extend({**row, "cohort": index} for row in cohort_loads)
        counts.append(len(cohort_loads))
        critical += schedule(cohort_loads, concurrent)
    evaluation.require(len(measured) == len(calls), "Incomplete evaluation measurement inventory")
    return {"log_sha256": digest, "measurements": tuple(measured), "loads": tuple(loads), "cohorts": len(counts),
            "sessions_per_cohort": tuple(counts), "concurrent": concurrent, "critical_path_seconds": critical,
            "equal_results": sum(call.result == first.result for call, first in zip(calls, expected, strict=True))}


def call_record(root, call, *, index, reported):
    fields(reported, "index binding exit_code process_seconds stderr_sha256 stdout_sha256 result_equal response_tokens load inference")
    evaluation.require(evaluation.natural(reported["index"]) == index and type(reported["exit_code"]) is int
                       and reported["exit_code"] == 0, "Failed or reordered direct process")
    evaluation.require(reported["binding"] == call.consumed["binding"], "Direct measurement binding mismatch")
    evaluation.require(type(reported["result_equal"]) is bool, "Invalid result equality observation")
    positive(reported["process_seconds"])
    prefix = root / f"{index:0{direct.INDEX_WIDTH}d}"
    actual = direct.inspect(Path(str(prefix) + ".stdout.jsonl"), call)
    for key, value in actual.items():
        evaluation.require(reported[key] == value, "Direct measurement differs from raw output")
    stderr_digest, _ = evaluation.snapshot(Path(str(prefix) + ".stderr.log"))
    evaluation.require(identity(reported["stderr_sha256"]) == stderr_digest, "Direct stderr identity mismatch")
    _, status = document(Path(str(prefix) + ".status.json"))
    fields(status, "index binding exit_code process_seconds stderr_sha256")
    evaluation.require(status == {key: reported[key] for key in status}, "Direct status differs from the measurement record")
    raw_digest, raw = records(Path(str(prefix) + ".stdout.jsonl"))
    evaluation.require(raw_digest == actual["stdout_sha256"], "Direct worker output changed during measurement")
    measured, loads = measured_stream(raw)
    evaluation.require(len(measured) == len(loads) == 1, "Expected one loaded session with one result in a process replay")
    evaluation.require(measured[0]["inference"] == actual["inference"]
                       and {key: loads[0][key] for key in actual["load"]} == actual["load"],
                       "Process measurements differ from the parsed session")
    return {"measurement": measured[0], "load": loads[0], "process_seconds": reported["process_seconds"],
            "result_equal": reported["result_equal"]}


def session_record(root, expected):
    log_digest, rows = records(root / "calls.jsonl")
    evaluation.require(len(rows) == len(expected), "Incomplete session call records")
    measured, loads, process_seconds, equal, position = [], [], 0.0, 0, 0
    for index, members in direct.cohorts(expected):
        prefix = root / f"session-{index:0{direct.INDEX_WIDTH}d}"
        _, status = document(Path(str(prefix) + ".status.json"))
        fields(status, "cohort exit_code process_seconds calls stderr_sha256")
        evaluation.require(status["cohort"] == index and type(status["exit_code"]) is int and status["exit_code"] == 0
                           and evaluation.natural(status["calls"]) == len(members), "Failed or incomplete session replay")
        process_seconds += positive(status["process_seconds"])
        stderr_digest, _ = evaluation.snapshot(Path(str(prefix) + ".stderr.log"))
        evaluation.require(identity(status["stderr_sha256"]) == stderr_digest, "Session stderr identity mismatch")
        actual = direct.inspect_session(Path(str(prefix) + ".stdout.jsonl"), members)
        for observed in actual["calls"]:
            reported = rows[position]
            position += 1
            fields(reported, "cohort index binding result_equal response_tokens inference")
            evaluation.require(reported == {**observed, "cohort": index}, "Session call record differs from raw output")
            equal += reported["result_equal"]
        raw_digest, raw = records(Path(str(prefix) + ".stdout.jsonl"))
        evaluation.require(raw_digest == actual["stdout_sha256"], "Session worker output changed during measurement")
        blocks, cohort_loads = measured_stream(raw)
        evaluation.require(len(cohort_loads) == 1 and len(blocks) == len(members),
                           "Expected one model load and one result per call in a session replay")
        for row, observed in zip(blocks, actual["calls"], strict=True):
            evaluation.require(row["inference"] == observed["inference"] and row["response_tokens"] == observed["response_tokens"],
                               "Session measurements differ from the parsed stream")
        evaluation.require({key: cohort_loads[0][key] for key in actual["load"]} == actual["load"],
                           "Session load measurement differs from the parsed stream")
        measured.extend(blocks)
        loads.append({**cohort_loads[0], "cohort": index})
    return {"measurements": tuple(measured), "loads": tuple(loads), "process_seconds": process_seconds, "log_sha256": log_digest,
            "equal_results": equal}


def direct_run(run, reference, expected, *, reference_digest, tasks_digest):
    root = Path(run["path"])
    complete_digest, complete = document(root / "complete.json")
    grouped = direct.cohorts(expected)
    declared = " mode cohorts loads" if any(key in complete for key in ("mode", "cohorts", "loads")) else ""
    fields(complete, "reference_log_sha256 tasks_sha256 policy calls wall_seconds process_seconds response_tokens equal_results scope" + declared)
    if declared:
        evaluation.require(complete["mode"] in direct.MODES, "Unknown direct replay mode")
        evaluation.require(evaluation.natural(complete["cohorts"]) == len(grouped), "Direct replay cohort count differs from the reference")
    else:
        complete = {**complete, "mode": "process", "cohorts": len(grouped), "loads": complete["calls"]}
    evaluation.require(complete["reference_log_sha256"] == reference_digest and complete["tasks_sha256"] == tasks_digest
                       and complete["policy"] == reference.policy, "Direct completion belongs to a different reference")
    positive(complete["wall_seconds"])
    evaluation.require(evaluation.natural(complete["calls"]) == len(expected), "Incomplete direct measurements")
    if complete["mode"] == "process":
        log_digest, rows = records(root / "calls.jsonl")
        evaluation.require(len(rows) == len(expected), "Incomplete direct measurements")
        observed = tuple(call_record(root, call, index=index, reported=record)
                         for index, (call, record) in enumerate(zip(expected, rows, strict=True)))
        measured = tuple(row["measurement"] for row in observed)
        loads = tuple({**row["load"], "cohort": call.cohort} for row, call in zip(observed, expected, strict=True))
        process_seconds = sum(row["process_seconds"] for row in observed)
        equal = sum(row["result_equal"] for row in observed)
    else:
        session = session_record(root, expected)
        measured, loads, log_digest = session["measurements"], session["loads"], session["log_sha256"]
        process_seconds, equal = session["process_seconds"], session["equal_results"]
    counts = tuple(sum(row["cohort"] == index for row in loads) for index, _ in grouped)
    totals = {"loads": len(loads), "process_seconds": process_seconds,
              "response_tokens": sum(row["response_tokens"] for row in measured), "equal_results": equal}
    for key, value in totals.items():
        evaluation.require(number(complete[key]) == value, "Direct completion differs from its call records")
    evaluation.require(complete["wall_seconds"] >= totals["process_seconds"], "Campaign duration is shorter than its subprocesses")
    return {"completion_sha256": complete_digest, "log_sha256": log_digest, "measurements": measured, "loads": loads,
            "cohorts": len(grouped), "sessions_per_cohort": counts, "concurrent": False,
            "critical_path_seconds": schedule(loads, False), "equal_results": equal,
            "campaign_wall_seconds": complete["wall_seconds"], "process_seconds": process_seconds}


def manifest(path):
    digest, value = document(path)
    fields(value, "reference_log reference_exit_code runs")
    evaluation.require(isinstance(value["reference_log"], str) and value["reference_log"], "Missing reference log path")
    evaluation.require(type(value["reference_exit_code"]) is int and value["reference_exit_code"] == 0,
                       "Reference evaluation did not exit successfully")
    evaluation.require(isinstance(value["runs"], list) and value["runs"], "Missing performance runs")
    for run in value["runs"]:
        fields(run, "name route path exit_code elapsed_seconds")
        evaluation.require(all(isinstance(run[key], str) and run[key] for key in ("name", "path")), "Missing run identity or path")
        evaluation.require(run["route"] in ("invar", "direct"), "Unknown measurement route")
        evaluation.require(type(run["exit_code"]) is int and run["exit_code"] == 0, "Measured process did not exit successfully")
        positive(run["elapsed_seconds"])
    evaluation.require(len({run["name"] for run in value["runs"]}) == len(value["runs"]), "Repeated measurement identity")
    evaluation.require(len({Path(run["path"]).resolve() for run in value["runs"]}) == len(value["runs"]),
                       "Repeated measurement path")
    return digest, value
