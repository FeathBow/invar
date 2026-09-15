import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from worker import core
from worker import continuation as redo_resident

INDEX_WIDTH = 4


@dataclass(frozen=True, kw_only=True)
class Options:
    python: str
    worker: Path
    cache: Path
    initial: Path
    reference: Path
    log: Path
    output: Path
    updates: int
    exit_code: int
    core: str = "invar"
    mode: str = "finite"
    worker_config: Path | None = None

    def worker_command(self):
        configuration = [] if self.worker_config is None else [f"--config={self.worker_config}"]
        return [self.python, str(self.worker), f"--cache={self.cache}", f"--reference={self.reference}", *configuration]


@dataclass(frozen=True, kw_only=True)
class Update:
    consumed: dict
    checkpoint: Path
    document: dict


@dataclass(frozen=True, kw_only=True)
class Services:
    run: object
    clock: object
    spawn: object = None
    environment: object = None


def updates(options):
    observed = core.invoke(["inspect", "update-calls", "--log", options.log, "--initial", options.initial,
                            "--updates", options.updates, "--exit-code", options.exit_code,
                            "--mode", options.mode], executable=options.core)
    planned = [Update(consumed=json.loads(item["consumed_json"]), checkpoint=Path(item["checkpoint"]), document=item)
               for item in observed["updates"]]
    return observed["reference_log_sha256"], planned, observed["terminal"]


def envelope(update):
    value = {"invocation": {key: update.consumed[key] for key in ("binding", "program")},
             "request": update.consumed["request"], "load": update.consumed["load"]}
    return (json.dumps(value, allow_nan=False) + "\n").encode()


def permission(update):
    value = {key: update.consumed[key] for key in ("binding", "program")}
    return (json.dumps(value, allow_nan=False) + "\n").encode()


def command(update, options, staged):
    return [*options.worker_command(), f"--checkpoint={update.checkpoint}", f"--output={staged}"]


def inspect(path, update, staged, *, exit_code, core_executable):
    return core.invoke(["inspect", "update-output", "--log", path, "--output", staged, "--exit-code", exit_code],
                       executable=core_executable, stdin=json.dumps(update.document, allow_nan=False))


def execute(update, options, *, services, index):
    output = options.output / f"{index:0{INDEX_WIDTH}d}.stdout.jsonl"
    error = options.output / f"{index:0{INDEX_WIDTH}d}.stderr.log"
    staged = options.output / f"{index:0{INDEX_WIDTH}d}.checkpoint"
    with output.open("xb") as stdout, error.open("xb") as stderr:
        started = services.clock()
        completed = services.run(command(update, options, staged), input=envelope(update) + permission(update), stdout=stdout, stderr=stderr)
        seconds = services.clock() - started
    record = {"index": index, "binding": update.consumed["binding"], "exit_code": completed.returncode,
              "process_seconds": seconds, "input_checkpoint": str(update.checkpoint),
              "stderr_sha256": hashlib.sha256(error.read_bytes()).hexdigest()}
    with (options.output / f"{index:0{INDEX_WIDTH}d}.status.json").open("x") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
    completed.check_returncode()
    return {**record, **inspect(output, update, staged, exit_code=completed.returncode, core_executable=options.core)}


def run(options, services):
    digest, planned, ending = updates(options)
    if options.mode in ("resident", "shared"):
        return redo_resident.run(options, services, planned=planned, reference=(digest, ending),
                                 envelope=envelope, permission=permission)
    options.output.mkdir()
    started = services.clock()
    results = []
    with (options.output / "calls.jsonl").open("x") as stream:
        for index, update in enumerate(planned):
            result = execute(update, options, services=services, index=index)
            results.append(result)
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
    report = {"reference_log_sha256": digest, "reference_exit_code": options.exit_code, "terminal": ending,
              "updates": len(results), "wall_seconds": services.clock() - started,
              "process_seconds": sum(row["process_seconds"] for row in results),
              "equal_results": sum(row["result_equal"] for row in results),
              "scope": "direct learning-worker replays of consumed update requests; no Invar rollout, admission, publication or successor consumption"}
    with (options.output / "complete.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    return report


def arguments():
    parser = argparse.ArgumentParser(description="Replay consumed update requests directly through the learning worker")
    parser.add_argument("--python", required=True)
    parser.add_argument("--core", default="invar", help="Core executable owning update replay admission and comparison")
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--mode", choices=("finite", "resident", "shared"), default="finite")
    parser.add_argument("--worker-config", type=Path)
    for name in ("worker", "cache", "initial", "reference", "log", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    values = vars(parser.parse_args())
    return Options(**values)


if __name__ == "__main__":
    print(json.dumps(run(arguments(), Services(run=subprocess.run, clock=time.perf_counter,
                                              spawn=subprocess.Popen, environment=dict(os.environ))), sort_keys=True))
