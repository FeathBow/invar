from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import ExitStack
import json

from worker import core
from worker import pipes as process


def placements(options, count, *, environment):
    devices = options.devices
    if devices is None:
        if count != 1:
            raise ValueError("Resident direct replay with multiple physical owners requires explicit --devices")
        return ((None, dict(environment)),)
    if len(devices) != count or any(not item for item in devices):
        raise ValueError("Resident direct devices must name a device for each reference physical owner")
    return tuple((device, {**environment, "CUDA_VISIBLE_DEVICES": device}) for device in devices)


def inspect(child, options, *, serialized):
    return core.invoke(["inspect", "replay-output", "--log", child.prefix.with_suffix(".stdout.jsonl"),
                        "--mode", "resident", "--owner", child.owner.session, "--exit-code", child.process.returncode],
                       executable=options.core,
                       stdin=json.dumps([serialized(call) for call in child.calls], allow_nan=False))


def execute(children, planned, *, services, origin, queued, grouped):
    pool = ThreadPoolExecutor(max_workers=len(children))
    try:
        intervals = []
        for cohort, _ in grouped(planned):
            started = services.clock() - origin
            groups = [(child, members) for child in children for index, members in grouped(child.calls) if index == cohort]
            futures = [pool.submit(process.group, child, members, queued=queued(members)) for child, members in groups]
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            for future in done:
                future.result()
            intervals.append({"cohort": cohort, "start_offset_seconds": started, "end_offset_seconds": services.clock() - origin})
        closing, completed = [], {}
        for child in reversed(children):
            started = services.clock() - origin
            process.close(child, groups=len(grouped(child.calls)))
            ended = services.clock() - origin
            completed[child.owner.session] = process.status(child, ended)
            closing.append({"owner": child.owner.session, "start_offset_seconds": started, "end_offset_seconds": ended})
        return intervals, closing, completed
    finally:
        # Unblock every owned reader before joining the executor after a fault.
        process.stop(children)
        pool.shutdown(wait=True)


def run(options, services, *, reference, planned, owners, command, queued, grouped, serialized):
    if services.spawn is None or services.environment is None:
        raise ValueError("Resident direct replay requires injected process creation and environment")
    environments = placements(options, len(owners), environment=services.environment)
    options.output.mkdir()
    origin = services.clock()
    children = []
    with ExitStack() as files:
        try:
            for (index, calls), (device, environment) in zip(owners, environments, strict=True):
                children.append(process.launch(index, calls, device=device, command=command, environment=environment,
                                               output=options.output, services=services, files=files, origin=origin))
            intervals, closing, completed = execute(children, planned, services=services, origin=origin, queued=queued,
                                                    grouped=grouped)
        finally:
            process.stop(children)
            process.finish(children, services=services, origin=origin)
    outputs = [inspect(child, options, serialized=serialized) for child in children]
    rows = [row for cohort, _ in grouped(planned) for output in outputs for row in output["calls"] if row["cohort"] == cohort]
    with (options.output / "calls.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    seconds = 0.0
    for child in children:
        seconds += completed[child.owner.session]["process_seconds"]
    report = {"reference_log_sha256": reference["reference_log_sha256"], "tasks_sha256": reference["tasks_sha256"],
              "policy": options.policy, "mode": "resident", "sessions": len(children), "calls": len(rows),
              "cohorts": len(intervals), "loads": sum(bool(child.calls) for child in children),
              "wall_seconds": services.clock() - origin, "process_seconds": seconds,
              "response_tokens": sum(row["response_tokens"] for row in rows),
              "equal_results": sum(row["result_equal"] for row in rows),
              "cohort_intervals": intervals, "close_intervals": closing,
              "scope": "direct resident replay of the reference physical owners and finite groups, with cohort barriers and ordered final closes; original raw output and supplied process statuses are checked offline; no Invar execution or publication authority"}
    with (options.output / "complete.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    return report
