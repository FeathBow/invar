from dataclasses import dataclass
import json
from pathlib import Path

from worker.cohort import fields
from worker.invocation import decode as invocation
from worker.cohort import unique
from worker.session import Call, Reference, decode as call, reference

FORMAT = "invar-inference-batch-v1"


@dataclass(frozen=True, kw_only=True)
class Batch:
    adapter: Path
    reference: Reference | None
    calls: tuple[Call, ...]


def decode(encoded):
    value = json.loads(encoded, object_pairs_hook=unique)
    fields(value, "format adapter reference calls")
    if value["format"] != FORMAT or not isinstance(value["adapter"], str) or not value["adapter"]:
        raise ValueError("Expected a finite inference batch and an adapter location")
    if not isinstance(value["calls"], list) or not value["calls"] or any(not isinstance(item, str) for item in value["calls"]):
        raise ValueError("A finite inference batch requires original encoded calls")
    calls = tuple(call(json.loads(item, object_pairs_hook=unique)) for item in value["calls"])
    for name in ("call", "attempt", "instance"):
        if len({getattr(item.invocation, name) for item in calls}) != len(calls):
            raise ValueError("Finite inference batch reuses a call, attempt or activation instance")
    return Batch(adapter=Path(value["adapter"]), reference=reference(value["reference"]), calls=calls)


def approve(invocations, *, source):
    value = json.loads(source.readline(), object_pairs_hook=unique)
    fields(value, "format permissions")
    if value["format"] != FORMAT or not isinstance(value["permissions"], list):
        raise ValueError("Expected finite inference batch permissions")
    if any(not isinstance(item, str) for item in value["permissions"]):
        raise ValueError("Batch permissions must retain original encoded invocations")
    received = tuple(invocation(json.loads(item, object_pairs_hook=unique)) for item in value["permissions"])
    if received != tuple(invocations):
        raise ValueError("Batch execution permissions differ from the consumed invocation inventory")


def capture(operation):
    lines = []
    operation(emit=lambda stage, values: lines.append(json.dumps({"stage": stage, **values}, allow_nan=False) + "\n"))
    return "".join(lines)


def serve(options, *, source, loader, execute, permission):
    batch = decode(source.readline())
    runtime = loader(options.cache, batch.adapter, expected=batch.calls[0].identities)
    execute(runtime, batch.calls, reference=batch.reference, approve=permission)
    if source.readline():
        raise ValueError("Input follows the completed finite inference batch")
