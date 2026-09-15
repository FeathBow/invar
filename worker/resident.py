from dataclasses import dataclass
import hashlib
import json

from worker.cohort import fields
from worker.invocation import decode as invocation
from worker.hf.session import unique

FORMAT = "invar-resident-v1"


@dataclass(frozen=True, kw_only=True)
class History:
    calls: frozenset[int] = frozenset()
    attempts: frozenset[int] = frozenset()
    instances: frozenset[int] = frozenset()
    groups: int = 0

    def advance(self, calls):
        observed = {name + "s": frozenset(getattr(call.invocation, name) for call in calls)
                    for name in ("call", "attempt", "instance")}
        if any(getattr(self, name).intersection(values) for name, values in observed.items()):
            raise ValueError("Resident execution reuses a historical call, attempt or activation instance")
        return History(**{name: getattr(self, name) | values for name, values in observed.items()}, groups=self.groups + 1)


def encoded(value):
    return json.dumps(value, allow_nan=False) + "\n"


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique)


@dataclass(frozen=True, kw_only=True)
class Owner:
    role: str
    session: int

    def __post_init__(self):
        if self.role not in ("inference", "learning", "shared") or type(self.session) is not int or self.session < 0:
            raise ValueError("Expected a resident role and nonnegative physical session")

    def value(self):
        return {"role": self.role, "session": self.session}

    def check(self, value):
        fields(value, "role session")
        if Owner(**value) != self:
            raise ValueError("Resident command names a different physical owner")


class Transcript:
    def __init__(self, output):
        self.output = output
        self.digest = hashlib.sha256()

    def begin(self):
        self.digest = hashlib.sha256()

    def emit(self, stage, values):
        raw = encoded({"stage": stage, **values})
        self.digest.update(raw.encode("utf-8"))
        self.output.write(raw)
        self.output.flush()

    def acknowledge(self, stage, values, *, measurement):
        self.output.write(encoded({"stage": stage, **values, "measurement": measurement}))
        self.output.flush()


def measured(stage, operation, *, measure):
    records = []
    measure(stage, operation, emit=lambda name, values: records.append(encoded({"stage": name, **values})))
    record, = records
    return record


def release(owner, loads, *, source, transcript, operation, measure):
    value = decode(source.readline())
    fields(value, "format owner action loads result_sha256")
    owner.check(value["owner"])
    if not isinstance(value["loads"], list):
        raise ValueError("Resident release requires original load invocations")
    received = tuple(invocation(item) for item in value["loads"])
    expected = {"format": FORMAT, "owner": owner.value(), "action": "release",
                "result_sha256": transcript.digest.hexdigest(),
                "loads": [{"binding": item.binding(), "program": item.program} for item in loads]}
    if value != expected or received != tuple(loads):
        raise ValueError("Resident release differs from the actual completed transcript or load inventory")
    measurement = measured("released", operation, measure=measure)
    transcript.acknowledge("released", {key: item for key, item in expected.items() if key != "action"},
                           measurement=measurement)


def closing(owner, value):
    fields(value, "format owner action")
    owner.check(value["owner"])
    if value != {"format": FORMAT, "owner": owner.value(), "action": "close"}:
        raise ValueError("Expected an explicit resident close command")


def close(owner, groups, *, source, transcript, operation, measure):
    measurement = measured("closed", operation, measure=measure)
    transcript.acknowledge("closed", {"format": FORMAT, "owner": owner.value(), "groups": groups},
                           measurement=measurement)
    if source.readline():
        raise ValueError("Input follows the acknowledged resident shutdown")
