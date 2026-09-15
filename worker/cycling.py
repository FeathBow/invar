from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import ExitStack
import json
from pathlib import Path

from worker import core
from worker import direct
from worker import pipes as process
from worker import redo
from worker import continuation as redo_resident

INDEX_WIDTH = 4
COPY_BYTES = 1024 * 1024


def key(owner):
    return owner["role"], owner["session"]


def append(output, value):
    output.write(process.encode(value))
    output.flush()


def segment(child, start, outputs):
    remaining = child.output.tell() - start
    with child.prefix.with_suffix(".stdout.jsonl").open("rb") as source:
        source.seek(start)
        while remaining:
            raw = source.read(min(COPY_BYTES, remaining))
            if not raw:
                raise ValueError("Actual cycle output ended before its completed group")
            for output in outputs:
                output.write(raw)
            remaining -= len(raw)
    for output in outputs:
        output.flush()


def launch(options, services, *, planned, placements, files, origin):
    children = []
    try:
        # Core close order is the reverse of physical owner creation order.
        for closing in reversed(planned["close"]):
            owner = closing["owner"]
            role, index = key(owner)
            device, environment = placements[index] if role == "inference" else (None, services.environment)
            groups = [group for cycle in planned["cycles"] for group in cycle["groups"] if group["owner"] == owner]
            calls = tuple(call for group in groups for call in (group["calls"] if group["kind"] == "inference" else [group]))
            output = options.output / role
            output.mkdir(exist_ok=True)
            children.append(process.launch(index, calls, device=device, command=options.command(role), environment=environment,
                                            output=output, services=services, files=files, origin=origin, role=role))
            with children[-1].prefix.with_suffix(".command.json").open("x") as command:
                json.dump(children[-1].process.args, command)
        return children
    except BaseException:
        process.stop(children)
        process.finish(children, services=services, origin=origin)
        raise


def inferences(children, selected, *, checkpoint, output, pool, transcript):
    groups = [group for group in selected["groups"] if group["kind"] == "inference"]
    positions = {key(group["owner"]): children[key(group["owner"])].output.tell() for group in groups}
    futures = []
    for group in groups:
        calls = direct.decode_calls(group["calls"])
        queued = direct.batch_input(calls, adapter=checkpoint / "adapter.safetensors")
        futures.append(pool.submit(process.group, children[key(group["owner"])], calls, queued=queued))
    done, _ = wait(futures, return_when=FIRST_EXCEPTION)
    for future in done:
        future.result()
    with output.open("xb") as recorded:
        for group in groups:
            owner = key(group["owner"])
            segment(children[owner], positions[owner], (recorded, transcript))


def prepare(options, selected, *, tasks, observed, flags):
    reference = selected["update"]
    path = options.output / f"cycle-{selected['index']:0{INDEX_WIDTH}d}.expected.json"
    with path.open("x") as output:
        json.dump(reference, output, allow_nan=False)
    consumed = core.decode(reference["consumed_json"])
    inputs = {**selected["settings"], "tasks": tasks, "cohort": selected["index"], "log": observed,
              "call": consumed["binding"]["call"], "expected": path}
    actual = core.invoke(["replay", "input", *flags(inputs)], executable=options.core)
    with path.with_suffix(".input.json").open("x") as output:
        json.dump(actual, output, allow_nan=False)
    if not actual["equal_to_reference"]:
        raise ValueError("Actual replay inference produces a different fixed numerical update input")
    return redo.Update(consumed={**consumed, "program": actual["program"], "request": actual["request"]},
                       checkpoint=Path(reference["checkpoint"]), document=reference)


def learning(children, selected, *, update, checkpoint, staged, transcript):
    group, = [group for group in selected["groups"] if group["kind"] == "learning"]
    child = children[key(group["owner"])]
    start = child.output.tell()
    result = redo_resident.group(child, update, checkpoint=checkpoint, output=staged,
                                 envelope=redo.envelope, permission=redo.permission)
    segment(child, start, (transcript,))
    return result


def policy(options, checkpoint, *, expected):
    selected = core.invoke(["policy", "inspect", "--checkpoint", checkpoint], executable=options.core)
    if selected != expected:
        raise ValueError("Replay policy description differs from its selected inference policy")
    return selected


def cycle(options, selected, *, children, services, pool, checkpoint, tasks, transcript, flags, origin, sessions):
    index = selected["index"]
    started = services.clock()
    description = policy(options, checkpoint, expected=selected["policy"])
    inferred = options.output / f"cycle-{index:0{INDEX_WIDTH}d}.inference.jsonl"
    inferences(children, selected, checkpoint=checkpoint, output=inferred, pool=pool, transcript=transcript)
    update = prepare(options, selected, tasks=tasks, observed=inferred, flags=flags)
    staging = f"staged{index + 1}"
    destination = f"generation{index + 1}"
    directory = options.output / "checkpoints"
    result = learning(children, selected, update=update, checkpoint=checkpoint, staged=directory / staging, transcript=transcript)
    successor = {**description, "adapter": result["adapter"]}
    receipt = core.invoke(["replay", "publish", *flags({"output": directory, "staging": staging,
                           "destination": destination, "publication": selected["publication"]["publication"]})],
                          executable=options.core, stdin=json.dumps(successor, allow_nan=False))
    with (options.output / f"cycle-{index:0{INDEX_WIDTH}d}.publication.json").open("x") as output:
        json.dump(receipt, output, allow_nan=False)
    policy(options, directory / destination, expected=successor)
    published = {**selected["publication"], "checkpoint": str(directory / destination),
                 "policy": result["adapter"], "learner": result["learner"]}
    append(transcript, published)
    seconds = services.clock() - started
    append(transcript, {"phase": "cycle", "index": index, "sessions": sessions, "seconds": seconds})
    return directory / destination, {"index": index, "start_offset_seconds": started - origin,
                                     "end_offset_seconds": services.clock() - origin, "seconds": seconds}


def execute(options, services, *, planned, tasks, placements, flags):
    (options.output / "checkpoints").mkdir()
    origin = services.clock()
    children = []
    intervals, statuses = [], []
    pool = ThreadPoolExecutor(max_workers=planned["sessions"])
    with ExitStack() as files:
        transcript = files.enter_context((options.output / "training.jsonl").open("xb"))
        try:
            children = launch(options, services, planned=planned, placements=placements, files=files, origin=origin)
            indexed = {key(child.owner.value()): child for child in children}
            checkpoint = options.initial
            for selected in planned["cycles"]:
                checkpoint, interval = cycle(options, selected, children=indexed, services=services, pool=pool,
                                             checkpoint=checkpoint, tasks=tasks, transcript=transcript, flags=flags,
                                             origin=origin, sessions=planned["sessions"])
                intervals.append(interval)
            for closing in planned["close"]:
                child = indexed[key(closing["owner"])]
                start = child.output.tell()
                process.close(child, groups=closing["groups"])
                segment(child, start, (transcript,))
                statuses.append(process.status(child, services.clock() - origin))
        finally:
            process.stop(children)
            pool.shutdown(wait=True)
            process.finish(children, services=services, origin=origin)
    return {"cycles": intervals, "sessions": len(children), "execution_seconds": services.clock() - origin,
            "process_seconds": sum(item["process_seconds"] for item in statuses), "physical_processes": statuses,
            "publications": len(intervals)}
