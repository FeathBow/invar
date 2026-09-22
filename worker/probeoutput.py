from dataclasses import dataclass
import json
from typing import Iterable

from worker.distribution import Snapshot


@dataclass(frozen=True, kw_only=True)
class Rows:
    snapshots: Iterable[Snapshot]


def row_chunks(rows):
    yield "["
    separator = ""
    for snapshot in rows.snapshots:
        yield separator
        yield json.dumps({"step": snapshot.step, "probability_bits": snapshot.probability_bits}, allow_nan=False)
        separator = ", "
    yield "]"


def object_chunks(value):
    yield "{"
    separator = ""
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Probe object keys must be strings")
        yield separator
        yield json.dumps(key) + ": "
        yield from chunks(item)
        separator = ", "
    yield "}"


def chunks(value):
    if isinstance(value, Rows):
        yield from row_chunks(value)
    elif isinstance(value, dict):
        yield from object_chunks(value)
    else:
        yield json.dumps(value, allow_nan=False)


def encoded(value):
    yield from chunks(value)
    yield "\n"


def write(value, *, output):
    for part in encoded(value):
        output.write(part)
    output.flush()
