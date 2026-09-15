from contextlib import ExitStack
import hashlib
import json

from worker import core
from worker import pipes as process

INDEX_WIDTH = 4
FORMAT = "invar-learning-resident-v1"


def group(child, update, *, checkpoint, output, envelope, permission):
    digest = hashlib.sha256()
    process.send(child, process.encode({"format": FORMAT, "checkpoint": str(checkpoint), "output": str(output),
                                        "call": envelope(update).decode("utf-8")}))
    process.through(child, "consumed", allowed=("loading", "profile", "load", "activation", "loaded_learner",
                                               "probability_roles", "roles"), digest=digest)
    process.send(child, permission(update))
    result = process.through(child, "result", allowed=("reward_update",), digest=digest)
    process.release(child, [update.consumed["load"]], digest=digest)
    return result


def execute(child, options, services, *, planned, origin, envelope, permission):
    checkpoint = planned[0].checkpoint
    intervals = []
    for index, update in enumerate(planned):
        staged = options.output / f"{index:0{INDEX_WIDTH}d}.checkpoint"
        started = services.clock() - origin
        group(child, update, checkpoint=checkpoint, output=staged, envelope=envelope, permission=permission)
        intervals.append({"index": index, "input_checkpoint": str(checkpoint),
                          "start_offset_seconds": started, "end_offset_seconds": services.clock() - origin})
        checkpoint = staged
    process.close(child, groups=len(planned))
    return intervals, process.status(child, services.clock() - origin)


def inspect(child, options, planned):
    return core.invoke(["inspect", "update-output", "--mode", options.mode, "--log",
                        child.prefix.with_suffix(".stdout.jsonl"), "--output", options.output,
                        "--exit-code", child.process.returncode], executable=options.core,
                       stdin=json.dumps([update.document for update in planned], allow_nan=False))


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def run(options, services, *, planned, reference, envelope, permission):
    if services.spawn is None or services.environment is None:
        raise ValueError("Resident update replay requires injected process creation and environment")
    options.output.mkdir()
    origin = services.clock()
    children = []
    command = options.worker_command()
    role = "shared" if options.mode == "shared" else "learning"
    if options.mode == "shared":
        command.append("--shared")
    with ExitStack() as files:
        try:
            child = process.launch(0, tuple(planned), device=None, command=command, environment=services.environment,
                                   output=options.output, services=services, files=files, origin=origin, role=role)
            children.append(child)
            intervals, status = execute(child, options, services, planned=planned, origin=origin,
                                        envelope=envelope, permission=permission)
        finally:
            process.stop(children)
            process.finish(children, services=services, origin=origin)
    observed = inspect(child, options, planned)
    write(options.output / "observations.json", observed)
    rows = [{**result, **interval} for result, interval in zip(observed["calls"], intervals, strict=True)]
    with (options.output / "calls.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    report = {"reference_log_sha256": reference[0], "reference_exit_code": options.exit_code, "terminal": reference[1],
              "mode": options.mode, "sessions": 1, "loads": observed["loads"], "updates": len(rows),
              "wall_seconds": services.clock() - origin, "process_seconds": status["process_seconds"],
              "equal_results": sum(row["result_equal"] for row in rows),
              "scope": "direct resident learning-worker replay retaining live optimizer and RNG across original consumed requests, with release and final close; output is checked offline; no Invar rollout, publication or execution authority"}
    write(options.output / "complete.json", report)
    return report
