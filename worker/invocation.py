import json
import sys
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Invocation:
    call: int
    attempt: int
    instance: int
    program: str

    def binding(self):
        return {"call": self.call, "attempt": self.attempt, "instance": self.instance}


def decode(value):
    if not isinstance(value, dict) or set(value) != {"binding", "program"}:
        raise ValueError("Expected an invocation binding and exact program text")
    bound = value["binding"]
    if not isinstance(bound, dict) or set(bound) != {"call", "attempt", "instance"}:
        raise ValueError("Expected call, attempt and instance identities")
    if any(type(item) is not int or item < 0 for item in bound.values()):
        raise ValueError("Invocation identities must be nonnegative integers")
    if not isinstance(value["program"], str) or not value["program"]:
        raise ValueError("Expected nonempty program text")
    return Invocation(**bound, program=value["program"])


def read(*, source=None):
    return decode(json.loads((sys.stdin if source is None else source).readline()))


def approve(value, *, source=None):
    received = read(source=source)
    if received != value:
        raise ValueError("Execution permission differs from the consumed invocation")


def request(value):
    return {"prompt": value.prompt, "seed": value.seed,
            "temperature": value.temperature, "tokens": value.limit}
