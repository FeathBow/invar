import argparse
import hashlib
import json
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import evaluation
from cohort import fields, number

FP32_WORD_LIMIT = 1 << 32
INDEX_WIDTH = 4
MODES = ("process", "session")
SESSION_FIELDS = "binding program load adapter tokenizer base assembly request"


@dataclass(frozen=True, kw_only=True)
class Options:
    python: str
    worker: Path
    cache: Path
    adapter: Path
    policy: str
    tasks: Path
    reference_log: Path
    reference_exit_code: int
    output: Path
    mode: str


@dataclass(frozen=True, kw_only=True)
class Call:
    consumed: dict
    result: dict
    cohort: int = 0


@dataclass(frozen=True, kw_only=True)
class Services:
    run: object
    clock: object


def calls(options):
    expected = evaluation.tasks(options.tasks)
    run = evaluation.Run(log=options.reference_log, policy=options.policy, exit_code=options.reference_exit_code)
    digest, samples = evaluation.read(run, expected)
    observed_digest, encoded = evaluation.snapshot(options.reference_log)
    evaluation.require(digest == observed_digest, "Reference log changed during preparation")
    consumed, results, declared = reports(encoded)
    evaluation.require(len(consumed) == len(results) == len(samples), "Incomplete reference worker inventory")
    planned = tuple(Call(consumed=first, result=last, cohort=declared[evaluation.binding(first["binding"])][0])
                    for first, last in zip(consumed, results, strict=True))
    for call in planned:
        check_reference(call, declared, expected)
    evaluation.require(len({evaluation.binding(call.consumed["binding"]) for call in planned}) == len(samples),
                       "Repeated reference consumption binding")
    return digest, expected.digest, planned


def reports(encoded):
    records = tuple(evaluation.decode(line) for line in encoded.splitlines())
    consumed = tuple(row for row in records if row.get("stage") == "consumed")
    results = tuple(row for row in records if row.get("stage") == "result")
    declared = {evaluation.binding(item["binding"]): (row["cohort"], item["name"])
                for row in records if row.get("phase") == "evaluation" for item in row["samples"]}
    return consumed, results, declared


def check_reference(call, declared, expected):
    value = call.consumed
    materialization = evaluation.model_binding(value)
    loading = ' load' if 'load' in value else ''
    fields(value, "stage binding program adapter request " + " ".join(materialization) + loading)
    if loading:
        from invocation import decode

        load = decode(value['load'])
        evaluation.require(load.binding() == value['binding'], 'Reference load and inference bindings differ')
    evaluation.require(isinstance(value["program"], str) and value["program"], "Missing consumed program")
    bound = evaluation.binding(value["binding"])
    cohort, name = declared[bound]
    task = next(task for task in expected.cohorts[cohort] if task["name"] == name)
    request = {key: task[key] for key in ("prompt", "seed", "tokens", "temperature")}
    evaluation.require(value["request"] == request, "Reference consumption differs from the frozen task")
    for key in ("binding", "adapter", "request"):
        evaluation.require(call.result[key] == value[key], "Reference result differs from consumption")
    evaluation.require(evaluation.model_binding(call.result) == materialization, "Reference model and tokenizer differ from consumption")
    result_tokens(call.result)


def command(call, options):
    request = call.consumed["request"]
    materialization = [f"--{name}-digest={value}" for name, value in evaluation.model_binding(call.consumed).items()]
    return [options.python, str(options.worker), f"--cache={options.cache}", f"--adapter={options.adapter}",
            f"--digest={options.policy}", *materialization,
            *(f"--{key}={request[key]}" for key in ("prompt", "tokens", "temperature", "seed"))]


def envelope(call):
    value = {key: call.consumed[key] for key in ("binding", "program")}
    if 'load' in call.consumed:
        value = {**value, 'load': call.consumed['load']}
    return (json.dumps(value, allow_nan=False) + "\n").encode()


def permission(call):
    value = {key: call.consumed[key] for key in ('binding', 'program')}
    return (json.dumps(value, allow_nan=False) + '\n').encode()


def session_envelope(call):
    value = {key: item for key, item in call.consumed.items() if key != "stage"}
    evaluation.require(set(value) == set(SESSION_FIELDS.split()),
                       "A session replay requires a batch reference with load and materialization bindings")
    return (json.dumps(value, allow_nan=False) + "\n").encode()


def session_command(options):
    return [options.python, str(options.worker), f"--cache={options.cache}", f"--adapter={options.adapter}"]


def session_input(planned):
    return b"".join(session_envelope(call) + permission(call) for call in planned)


def measured(record):
    fields(record, "stage seconds peak_allocated peak_reserved")
    seconds = number(record["seconds"])
    evaluation.require(seconds >= 0, "Negative worker duration")
    return {"seconds": seconds, "peak_allocated": evaluation.natural(record["peak_allocated"]),
            "peak_reserved": evaluation.natural(record["peak_reserved"])}


def result_tokens(value):
    materialization = evaluation.model_binding(value)
    fields(value, "stage binding adapter request tokens prompt_length behavior text behavior_bits truncated"
           + " " + " ".join(materialization))
    tokens, bits, behavior = (value[key] for key in ("tokens", "behavior_bits", "behavior"))
    evaluation.require(all(isinstance(items, list) for items in (tokens, bits, behavior)), "Expected numerical sequences")
    prefix = evaluation.natural(value["prompt_length"])
    evaluation.require(0 < prefix < len(tokens), "Invalid prompt boundary")
    count = len(tokens) - prefix
    evaluation.require(len(bits) == len(behavior) == count <= value["request"]["tokens"], "Invalid response lengths")
    evaluation.require(all(evaluation.natural(token) >= 0 for token in tokens), "Invalid token identity")
    for probability, word in zip(behavior, bits, strict=True):
        behavior_word(probability, word)
    evaluation.require(isinstance(value["text"], str) and type(value["truncated"]) is bool, "Invalid decoded output")
    evaluation.require(not value["truncated"] or count == value["request"]["tokens"], "Invalid truncation length")
    return count


def behavior_word(value, word):
    evaluation.require(evaluation.natural(word) < FP32_WORD_LIMIT, "Invalid FP32 word")
    actual = number(struct.unpack("!f", struct.pack("!I", word))[0])
    reported = number(value)
    evaluation.require(actual <= 0 and reported <= 0, "Invalid behavior probability")
    evaluation.require(struct.pack("!d", actual) == struct.pack("!d", reported),
                       "Behavior value and FP32 word disagree")


def inspect(path, call):
    digest, encoded = evaluation.snapshot(path)
    evaluation.require(encoded.endswith(b"\n"), "Incomplete direct worker output")
    records = tuple(evaluation.decode(line) for line in encoded.splitlines())
    evaluation.require(all(isinstance(row, dict) and row.get("stage") in evaluation.WORKER_STAGES for row in records),
                       "Unexpected direct worker record")
    selected = {stage: tuple(row for row in records if row["stage"] == stage)
                for stage in ("consumed", "result", "load", "inference")}
    evaluation.require(all(len(rows) == 1 for rows in selected.values()), "Incomplete or repeated direct worker stages")
    evaluation.require(selected["consumed"][0] == call.consumed, "Direct worker consumed different input")
    result = selected["result"][0]
    evaluation.require(records[-1] == result, "Trailing direct worker output")
    for key in ("binding", "adapter", "request"):
        evaluation.require(result[key] == call.result[key], "Direct result binding differs from the reference")
    evaluation.require(evaluation.model_binding(result) == evaluation.model_binding(call.result),
                       "Direct model and tokenizer binding differs from the reference")
    return {"stdout_sha256": digest, "result_equal": result == call.result, "response_tokens": result_tokens(result),
            "load": measured(selected["load"][0]), "inference": measured(selected["inference"][0])}


def inspect_session(path, planned):
    digest, encoded = evaluation.snapshot(path)
    evaluation.require(encoded.endswith(b"\n"), "Incomplete direct worker output")
    records = tuple(evaluation.decode(line) for line in encoded.splitlines())
    evaluation.require(all(isinstance(row, dict) and row.get("stage") in evaluation.WORKER_STAGES for row in records),
                       "Unexpected direct worker record")
    positions = {stage: tuple(index for index, row in enumerate(records) if row["stage"] == stage)
                 for stage in ("load", "consumed", "inference", "result")}
    evaluation.require(len(positions["load"]) == 1, "Expected exactly one model load in a session replay")
    evaluation.require(all(len(positions[stage]) == len(planned) for stage in ("consumed", "inference", "result")),
                       "Incomplete or repeated session replay stages")
    evaluation.require(records[-1] is records[positions["result"][-1]], "Trailing direct worker output")
    previous = positions["load"][0]
    calls = []
    for index, call in enumerate(planned):
        first, timing, last = (records[positions[stage][index]] for stage in ("consumed", "inference", "result"))
        evaluation.require(previous < positions["consumed"][index] < positions["inference"][index] < positions["result"][index],
                           "Reordered session replay stages")
        previous = positions["result"][index]
        evaluation.require(first == call.consumed, "Direct worker consumed different input")
        for key in ("binding", "adapter", "request"):
            evaluation.require(last[key] == call.result[key], "Direct result binding differs from the reference")
        evaluation.require(evaluation.model_binding(last) == evaluation.model_binding(call.result),
                           "Direct model and tokenizer binding differs from the reference")
        calls.append({"index": index, "binding": call.consumed["binding"], "result_equal": last == call.result,
                      "response_tokens": result_tokens(last), "inference": measured(timing)})
    return {"stdout_sha256": digest, "load": measured(records[positions["load"][0]]), "calls": calls}


def cohorts(planned):
    grouped = {}
    for call in planned:
        grouped.setdefault(call.cohort, []).append(call)
    return tuple(grouped.items())


def execute_session(planned, options, *, services, queued, index):
    output = options.output / f"session-{index:0{INDEX_WIDTH}d}.stdout.jsonl"
    error = options.output / f"session-{index:0{INDEX_WIDTH}d}.stderr.log"
    with output.open("xb") as stdout, error.open("xb") as stderr:
        started = services.clock()
        completed = services.run(session_command(options), input=queued, stdout=stdout, stderr=stderr)
        seconds = services.clock() - started
    record = {"cohort": index, "exit_code": completed.returncode, "process_seconds": seconds, "calls": len(planned),
              "stderr_sha256": hashlib.sha256(error.read_bytes()).hexdigest()}
    with (options.output / f"session-{index:0{INDEX_WIDTH}d}.status.json").open("x") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
    completed.check_returncode()
    return {**record, **inspect_session(output, planned)}


def execute(call, options, *, services, index):
    output = options.output / f"{index:0{INDEX_WIDTH}d}.stdout.jsonl"
    error = options.output / f"{index:0{INDEX_WIDTH}d}.stderr.log"
    with output.open("xb") as stdout, error.open("xb") as stderr:
        started = services.clock()
        completed = services.run(command(call, options), input=envelope(call) + permission(call), stdout=stdout, stderr=stderr)
        seconds = services.clock() - started
    record = {"index": index, "binding": call.consumed["binding"], "exit_code": completed.returncode,
              "process_seconds": seconds, "stderr_sha256": hashlib.sha256(error.read_bytes()).hexdigest()}
    with (options.output / f"{index:0{INDEX_WIDTH}d}.status.json").open("x") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
    completed.check_returncode()
    return {**record, **inspect(output, call)}


def run(options, services):
    evaluation.require(options.mode in MODES, "Unknown direct replay mode")
    digest, tasks_digest, planned = calls(options)
    evaluation.require(all(call.consumed["adapter"] == options.policy for call in planned), "Reference policy mismatch")
    grouped = cohorts(planned)
    queued = {index: session_input(members) for index, members in grouped} if options.mode == "session" else None
    options.output.mkdir()
    started = services.clock()
    results = []
    if options.mode == "session":
        process_seconds = 0.0
        with (options.output / "calls.jsonl").open("x") as stream:
            for index, members in grouped:
                observed = execute_session(members, options, services=services, queued=queued[index], index=index)
                process_seconds += observed["process_seconds"]
                for result in observed["calls"]:
                    result = {**result, "cohort": index}
                    results.append(result)
                    stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
        loads = len(grouped)
        scope = "direct session replays, one process and one model load per cohort in sequence; no Invar admission, completion or publication evidence"
    else:
        with (options.output / "calls.jsonl").open("x") as stream:
            for index, call in enumerate(planned):
                result = execute(call, options, services=services, index=index)
                results.append(result)
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
        loads, process_seconds = len(results), sum(row["process_seconds"] for row in results)
        scope = "direct worker measurements with one process and one model load per request in sequence; no Invar admission, completion or publication evidence"
    report = {"reference_log_sha256": digest, "tasks_sha256": tasks_digest,
              "policy": options.policy, "mode": options.mode, "calls": len(results), "cohorts": len(grouped), "loads": loads,
              "wall_seconds": services.clock() - started, "process_seconds": process_seconds,
              "response_tokens": sum(row["response_tokens"] for row in results),
              "equal_results": sum(row["result_equal"] for row in results), "scope": scope}
    with (options.output / "complete.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    return report


def arguments():
    parser = argparse.ArgumentParser(description="Replay all consumed requests from one complete evaluation directly")
    parser.add_argument("--python", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--reference-exit-code", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    for name in ("worker", "cache", "adapter", "tasks", "reference-log", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return Options(**vars(parser.parse_args()))


if __name__ == "__main__":
    print(json.dumps(run(arguments(), Services(run=subprocess.run, clock=time.perf_counter)), sort_keys=True))
