import hashlib
import json
import subprocess
from dataclasses import dataclass

from worker import core
from worker.resident import FORMAT, Owner

INDEX_WIDTH = 4


@dataclass(frozen=True, kw_only=True)
class Child:
    owner: Owner
    calls: tuple
    device: str | None
    process: object
    output: object
    prefix: object
    started: float


def launch(index, calls, *, device, command, environment, output, services, files, origin, role="inference"):
    prefix = output / f"owner-{index:0{INDEX_WIDTH}d}"
    stdout = files.enter_context(prefix.with_suffix(".stdout.jsonl").open("xb"))
    stderr = files.enter_context(prefix.with_suffix(".stderr.log").open("xb"))
    started = services.clock() - origin
    process = services.spawn([*command, f"--session={index}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=stderr, env=environment)
    return Child(owner=Owner(role=role, session=index), calls=calls, device=device,
                 process=process, output=stdout, prefix=prefix, started=started)


def send(child, raw):
    child.process.stdin.write(raw)
    child.process.stdin.flush()


def encode(value):
    return (json.dumps(value, allow_nan=False) + "\n").encode()


def receive(child, *, digest=None):
    raw = child.process.stdout.readline()
    child.output.write(raw)
    child.output.flush()
    if not raw.endswith(b"\n"):
        raise ValueError("Resident direct worker ended an incomplete exchange")
    if digest is not None:
        digest.update(raw)
    value = core.decode(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a resident direct worker object")
    return value


def through(child, expected, *, allowed, digest):
    while True:
        value = receive(child, digest=digest)
        stage = value.get("stage")
        if stage == expected:
            return value
        if stage not in allowed:
            raise ValueError(f"Unexpected resident direct output before {expected}: {stage}")


def group(child, calls, *, queued):
    prepared, permitted = queued.splitlines(keepends=True)
    digest = hashlib.sha256()
    send(child, prepared)
    through(child, "consumed", allowed=("loading", "profile", "load", "activation"), digest=digest)
    send(child, permitted)
    through(child, "result", allowed=("inference",), digest=digest)
    release(child, [call.consumed["load"] for call in calls], digest=digest)


def release(child, loads, *, digest):
    released = {"format": FORMAT, "owner": child.owner.value(), "loads": loads,
                "result_sha256": digest.hexdigest()}
    send(child, encode({"action": "release", **released}))
    actual = receive(child)
    measurement = actual.get("measurement")
    if not isinstance(measurement, str) or actual != {"stage": "released", **released, "measurement": measurement}:
        raise ValueError("Resident direct release differs from its original output or load inventory")


def close(child, *, groups):
    send(child, encode({"format": FORMAT, "owner": child.owner.value(), "action": "close"}))
    child.process.stdin.close()
    actual = receive(child)
    measurement = actual.get("measurement")
    if not isinstance(measurement, str) or actual != {"stage": "closed", "format": FORMAT, "owner": child.owner.value(),
                                                   "groups": groups, "measurement": measurement}:
        raise ValueError("Resident direct close differs from its complete physical lifetime")
    trailing = drain(child)
    code = child.process.wait()
    if code:
        raise subprocess.CalledProcessError(code, child.process.args)
    if trailing:
        raise ValueError("Output follows the final resident direct close")


def drain(child):
    trailing = child.process.stdout.read()
    child.output.write(trailing)
    child.output.flush()
    return bool(trailing)


def status(child, ended):
    record = {"owner": child.owner.session, "pid": child.process.pid, "device": child.device,
              "calls": len(child.calls), "exit_code": child.process.returncode,
              "start_offset_seconds": child.started, "end_offset_seconds": ended,
              "process_seconds": ended - child.started,
              **{name + "_sha256": hashlib.sha256(child.prefix.with_suffix(suffix).read_bytes()).hexdigest()
                 for name, suffix in (("stdout", ".stdout.jsonl"), ("stderr", ".stderr.log"))}}
    with child.prefix.with_suffix(".status.json").open("x") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
    return record


def stop(children):
    for child in children:
        if child.process.poll() is None:
            child.process.kill()


def finish(children, *, services, origin):
    failures = []
    for child in children:
        operations = [lambda: drain(child), child.process.wait]
        if not child.prefix.with_suffix(".status.json").exists():
            operations.append(lambda: status(child, services.clock() - origin))
        operations.extend((child.process.stdout.close, child.process.stdin.close))
        for operation in operations:
            try:
                operation()
            except BaseException as error:
                failures.append(error)
    if failures:
        raise BaseExceptionGroup("Resident direct child cleanup failed", failures)
