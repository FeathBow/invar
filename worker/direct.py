import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from worker import core
from worker import partition as direct_resident
from worker.batch import FORMAT as BATCH_FORMAT

INDEX_WIDTH = 4
MODES = ("process", "session", "batch", "resident")
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
    core: str = "invar"
    worker_config: Path | None = None
    devices: tuple[str, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class Call:
    consumed: dict
    result: dict
    cohort: int = 0


@dataclass(frozen=True, kw_only=True)
class Services:
    run: object
    clock: object
    spawn: object = None
    environment: object = None


def calls(options, *, mode="process", core_executable="invar"):
    observed = reference(options, mode=mode, core_executable=core_executable)
    return observed["reference_log_sha256"], observed["tasks_sha256"], decode_calls(observed["calls"])


def reference(options, *, mode, core_executable):
    return core.invoke(["inspect", "replay-calls", "--tasks", options.tasks, "--log", options.reference_log,
                            "--policy", options.policy, "--exit-code", options.reference_exit_code, "--mode", mode],
                           executable=core_executable)


def decode_calls(rows):
    return tuple(Call(consumed=core.decode(row["consumed_json"]), result=core.decode(row["result_json"]),
                      cohort=row["cohort"]) for row in rows)


def command(call, options):
    request = call.consumed["request"]
    materialization = [f"--{name}-digest={call.consumed[name]}" for name in ("tokenizer", "base", "assembly") if name in call.consumed]
    return [options.python, str(options.worker), f"--cache={options.cache}", f"--adapter={options.adapter}",
            f"--digest={options.policy}", *configuration(options), *materialization,
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
    return (json.dumps(value, allow_nan=False) + "\n").encode()


def session_command(options):
    adapter = [] if options.mode in ("batch", "resident") else [f"--adapter={options.adapter}"]
    return [options.python, str(options.worker), f"--cache={options.cache}", *adapter, *configuration(options)]


def configuration(options):
    return [] if options.worker_config is None else [f"--config={options.worker_config}"]


def session_input(planned):
    return b"".join(session_envelope(call) + permission(call) for call in planned)


def batch_input(planned, *, adapter):
    frames = ({"format": BATCH_FORMAT, "adapter": str(adapter),
               "calls": [session_envelope(call).decode() for call in planned]},
              {"format": BATCH_FORMAT, "permissions": [permission(call).decode() for call in planned]})
    return "".join(json.dumps(frame, allow_nan=False) + "\n" for frame in frames).encode()


def serialized(call):
    return {"cohort": call.cohort, "consumed_json": json.dumps(call.consumed, allow_nan=False),
            "result_json": json.dumps(call.result, allow_nan=False)}


def inspect(path, call, *, exit_code, core_executable="invar"):
    return core.invoke(["inspect", "replay-output", "--log", path, "--mode", "process", "--exit-code", exit_code],
                       executable=core_executable, stdin=json.dumps([serialized(call)], allow_nan=False))


def inspect_session(path, planned, *, exit_code, core_executable="invar", mode="session"):
    return core.invoke(["inspect", "replay-output", "--log", path, "--mode", mode, "--exit-code", exit_code],
                       executable=core_executable, stdin=json.dumps([serialized(call) for call in planned], allow_nan=False))


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
    return {**record, **inspect_session(output, planned, exit_code=completed.returncode,
                                      core_executable=options.core, mode=options.mode)}


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
    return {**record, **inspect(output, call, exit_code=completed.returncode, core_executable=options.core)}


def run(options, services):
    if options.mode == "resident":
        return resident(options, services, runner=direct_resident.run)
    if options.devices is not None:
        raise ValueError("Direct --devices requires resident mode")
    return finite(options, services)


def resident_execute(options, services):
    if options.mode != "resident":
        raise ValueError("Execution-only direct inference requires resident mode")
    return resident(options, services, runner=direct_resident.run_execution)


def resident(options, services, *, runner):
    observed = reference(options, mode=options.mode, core_executable=options.core)
    owners = tuple((owner["owner"], decode_calls(owner["calls"])) for owner in observed["residence"]["owners"])
    return runner(options, services, reference=observed, planned=decode_calls(observed["calls"]),
                  owners=owners, command=session_command(options), grouped=cohorts, serialized=serialized,
                  queued=partial(batch_input, adapter=options.adapter))


def finite(options, services):
    digest, tasks_digest, planned = calls(options, mode=options.mode, core_executable=options.core)
    grouped = cohorts(planned)
    queued = {index: batch_input(members, adapter=options.adapter) if options.mode == "batch" else session_input(members)
              for index, members in grouped} if options.mode != "process" else None
    options.output.mkdir()
    started = services.clock()
    results = []
    if options.mode != "process":
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
        scope = f"direct {options.mode} replays, one process and one model load per cohort in sequence; no Invar admission, completion or publication evidence"
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
    parser.add_argument("--core", default="invar", help="Core executable owning replay admission")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--reference-exit-code", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--worker-config", type=Path, help="Explicit configuration passed to the selected worker")
    parser.add_argument("--devices", type=lambda value: tuple(value.split(",")),
                        help="Resident physical owner CUDA devices in reference order, required for multiple owners")
    for name in ("worker", "cache", "adapter", "tasks", "reference-log", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return Options(**vars(parser.parse_args()))


def main(*, execution_only=False):
    selected = resident_execute if execution_only else run
    print(json.dumps(selected(arguments(), Services(run=subprocess.run, clock=time.perf_counter, spawn=subprocess.Popen,
                                                    environment=dict(os.environ))), sort_keys=True))


if __name__ == "__main__":
    main()
