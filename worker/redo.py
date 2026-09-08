import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import evaluation
from session import unique

INDEX_WIDTH = 4
HEADER_SIZE = 8
RESULT_FIELDS = ("adapter", "learner", "probabilities", "update")
MEASURED = ("load", "probability_roles", "reward_update")


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


@dataclass(frozen=True, kw_only=True)
class Update:
    consumed: dict
    result: dict
    checkpoint: Path
    published: Path


@dataclass(frozen=True, kw_only=True)
class Services:
    run: object
    clock: object


def records(path):
    encoded = path.read_bytes()
    evaluation.require(encoded.endswith(b"\n"), "Incomplete training stream")
    return hashlib.sha256(encoded).hexdigest(), [evaluation.decode(line) for line in encoded.splitlines()]


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal(rows, expected):
    cycles = [row for row in rows if row.get("phase") == "cycle"]
    if not cycles:
        return "publication count and exit status"
    evaluation.require(len(cycles) == expected and [evaluation.natural(row["index"]) for row in cycles] == list(range(expected)),
                       "Training stream cycle records differ from the expected updates")
    evaluation.require(rows[-1] is cycles[-1], "Training stream does not end with its final cycle record")
    return "cycle records"


def updates(rows, initial, expected, exit_code):
    evaluation.require(type(exit_code) is int and exit_code == 0, "Training process did not exit successfully")
    evaluation.require(type(expected) is int and expected > 0, "Expected a positive update count")
    consumed = [row for row in rows if row.get("stage") == "consumed" and "samples" in row.get("request", {})]
    results = {json.dumps(row["binding"], sort_keys=True): row for row in rows if row.get("stage") == "result" and "update" in row}
    published = [row for row in rows if row.get("phase") == "published"]
    evaluation.require(len(consumed) == len(results) == len(published) == expected,
                       "Training stream does not contain exactly the expected consumed, result and published records")
    ending = terminal(rows, expected)
    positions = {id(row): position for position, row in enumerate(rows)}
    cycles = [row for row in rows if row.get("phase") == "cycle"]
    evaluation.require(all(positions[id(published[index])] < positions[id(cycle)] for index, cycle in enumerate(cycles)),
                       "Cycle record precedes its publication")
    planned = []
    for index, row in enumerate(consumed):
        result = results.get(json.dumps(row["binding"], sort_keys=True))
        evaluation.require(result is not None, "Update result missing for a consumed request")
        evaluation.require(published[index]["binding"] == row["binding"], "Publication order differs from consumption order")
        evaluation.require(result["adapter"] == published[index]["policy"] and result["learner"] == published[index]["learner"], "Published identities differ from the update result")
        checkpoint = initial if index == 0 else Path(published[index - 1]["checkpoint"])
        evaluation.require(checkpoint.is_dir(), "Update input checkpoint is not a directory")
        planned.append(Update(consumed=row, result=result, checkpoint=checkpoint, published=Path(published[index]["checkpoint"])))
    return planned, ending


def envelope(update):
    value = {"invocation": {key: update.consumed[key] for key in ("binding", "program")},
             "request": update.consumed["request"], "load": update.consumed["load"]}
    return (json.dumps(value, allow_nan=False) + "\n").encode()


def permission(update):
    value = {key: update.consumed[key] for key in ("binding", "program")}
    return (json.dumps(value, allow_nan=False) + "\n").encode()


def command(update, options, staged):
    return [options.python, str(options.worker), f"--cache={options.cache}", f"--checkpoint={update.checkpoint}",
            f"--reference={options.reference}", f"--output={staged}"]


def tensors(path):
    encoded = path.read_bytes()
    size = int.from_bytes(encoded[:HEADER_SIZE], "little")
    header = json.loads(encoded[HEADER_SIZE:HEADER_SIZE + size], object_pairs_hook=unique)
    metadata = header.pop("__metadata__", {})
    return header, metadata, encoded[HEADER_SIZE + size:]


def gradients_equal(actual, expected):
    return tensors(actual) == tensors(expected)


def inspect(path, update, staged):
    rows = [evaluation.decode(line) for line in path.read_bytes().splitlines()]
    consumed = [row for row in rows if row.get("stage") == "consumed"]
    results = [row for row in rows if row.get("stage") == "result"]
    evaluation.require(len(consumed) == 1 and len(results) == 1, "Expected one consumed and one result report")
    evaluation.require(consumed[0]["binding"] == results[0]["binding"] == update.consumed["binding"], "Replayed binding differs")
    evaluation.require(consumed[0]["request"] == update.consumed["request"], "Replayed consumption differs from the reference")
    measured = {row["stage"]: {key: value for key, value in row.items() if key != "stage"} for row in rows if row.get("stage") in MEASURED}
    evaluation.require(set(measured) == set(MEASURED), "Expected one measurement per worker stage")
    for name, key in (("gradients.safetensors", "gradients"), ("probabilities.json", "probabilities")):
        evaluation.require(file_digest(staged / name) == results[0][key], f"Replayed {key} file differs from its reported digest")
        evaluation.require(file_digest(update.published / name) == update.result[key], f"Reference {key} file differs from its reported digest")
    equal = {field: results[0][field] == update.result[field] for field in RESULT_FIELDS}
    equal["gradients"] = gradients_equal(staged / "gradients.safetensors", update.published / "gradients.safetensors")
    return {"measured": measured, "result_equal": all(equal.values()), "equal_fields": equal,
            "gradients_file_digest_equal": results[0]["gradients"] == update.result["gradients"],
            "adapter": results[0]["adapter"], "learner": results[0]["learner"]}


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
    return {**record, **inspect(output, update, staged)}


def run(options, services):
    digest, rows = records(options.log)
    planned, ending = updates(rows, options.initial, options.updates, options.exit_code)
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
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    for name in ("worker", "cache", "initial", "reference", "log", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    values = vars(parser.parse_args())
    return Options(**values)


if __name__ == "__main__":
    print(json.dumps(run(arguments(), Services(run=subprocess.run, clock=time.perf_counter)), sort_keys=True))
