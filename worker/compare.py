import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from safetensors import deserialize

import cohort
import invocation

OBSERVATION = "objective and reward gradients before AdamW"
ROLES = ("objective", "reward")
HEADER_BYTES = 8


@dataclass(frozen=True, kw_only=True)
class Claim:
    invocation: invocation.Invocation
    request: str
    digest: str
    result: str


@dataclass(frozen=True, kw_only=True)
class Tensor:
    dtype: str
    shape: tuple[int, ...]
    data: bytes


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def logical_request(encoded):
    value = json.loads(encoded)
    samples = {item["sample"]: item for item in value["samples"]}
    return canonical({**value, "samples": [samples[name] for name in value["order"]]})


def reports(path, call):
    selected = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line, object_pairs_hook=unique)
            if not isinstance(event, dict):
                raise ValueError("Expected JSON objects in the execution log")
            if event.get("stage") not in ("consumed", "result"):
                continue
            bound = event.get("binding")
            if isinstance(bound, dict) and bound.get("call") == call:
                selected.append(event)
    return selected


def claim(path, call):
    if type(call) is not int or call < 0:
        raise ValueError("Expected a nonnegative update call")
    selected = reports(path, call)
    if [item["stage"] for item in selected] != ["consumed", "result"]:
        raise ValueError("Expected one consumed/result pair for the selected call")
    consumed, result = selected
    bound = invocation.decode({"binding": consumed["binding"], "program": consumed["program"]})
    if canonical(result["binding"]) != canonical(bound.binding()):
        raise ValueError("Update result binding differs from consumption")
    cohort.decode(consumed["request"])
    request = canonical(consumed["request"])
    if canonical(result["request"]) != request:
        raise ValueError("Update result input differs from consumption")
    return Claim(invocation=bound, request=request, digest=cohort.identity(result["gradients"]),
                 result=canonical(result))


def snapshot(path, expected):
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected.digest:
        raise ValueError("Gradient file differs from its reported digest")
    decoded = deserialize(encoded)
    length = int.from_bytes(encoded[:HEADER_BYTES], "little")
    header = json.loads(encoded[HEADER_BYTES:HEADER_BYTES + length], object_pairs_hook=unique)
    metadata(header.get("__metadata__"), expected)
    tensors = tuple(sorted((name, Tensor(dtype=value["dtype"], shape=tuple(value["shape"]),
                                        data=bytes(value["data"]))) for name, value in decoded))
    validate_roles(tensors)
    return tensors


def metadata(value, expected):
    if not isinstance(value, dict) or set(value) != {"binding", "program", "policy", "observation"}:
        raise ValueError("Expected the bound gradient observation metadata")
    if any(not isinstance(item, str) for item in value.values()):
        raise ValueError("Gradient metadata values must be strings")
    bound = invocation.decode({"binding": json.loads(value["binding"], object_pairs_hook=unique),
                               "program": value["program"]})
    if bound != expected.invocation:
        raise ValueError("Gradient metadata invocation differs from its report")
    if value["policy"] != json.loads(expected.request)["policy"]:
        raise ValueError("Gradient metadata policy differs from its consumed input")
    if value["observation"] != OBSERVATION:
        raise ValueError("Gradient observation is not the declared pre-AdamW snapshot")


def validate_roles(tensors):
    parameters = {role: set() for role in ROLES}
    for name, _ in tensors:
        role, separator, parameter = name.partition("/")
        if role not in parameters or not separator or not parameter:
            raise ValueError(f"Invalid gradient tensor role: {name}")
        parameters[role].add(parameter)
    if not parameters["objective"] or parameters["objective"] != parameters["reward"]:
        raise ValueError("Objective and reward snapshots must cover the same nonempty parameter set")


def differences(first, second):
    left, right = dict(first), dict(second)
    changes = []
    for name in sorted(left.keys() | right.keys()):
        if name not in left or name not in right:
            changes.append({"tensor": name, "missing": "left" if name not in left else "right"})
            continue
        fields = [field for field in ("dtype", "shape", "data")
                  if getattr(left[name], field) != getattr(right[name], field)]
        if fields:
            changes.append({"tensor": name, "fields": fields})
    return changes


def inventory(path, identity):
    from binding import schema
    from policy import read_adapter

    state = read_adapter(path, identity)
    expected = {}
    for parameter, tensor in schema(state).items():
        for role in ROLES:
            expected[f"{role}/{parameter}"] = tuple(tensor.shape)
    return expected


def complete(tensors, expected):
    received = dict(tensors)
    if received.keys() != expected.keys():
        missing = sorted(expected.keys() - received.keys())
        extra = sorted(received.keys() - expected.keys())
        raise ValueError(f"Gradient parameter inventory mismatch: missing={missing}, extra={extra}")
    for name, shape in expected.items():
        if received[name].dtype != "F32" or received[name].shape != shape:
            raise ValueError(f"Gradient metadata differs from the input adapter: {name}")


def paired(left, right):
    first = claim(left.log, left.call)
    second = claim(right.log, right.call)
    if (first.invocation.program != second.invocation.program
            or logical_request(first.request) != logical_request(second.request)):
        raise ValueError("Compared updates must have the same program and consumed numerical input")
    return first, second


def compare(left, right, policy):
    first, second = paired(left, right)
    expected = inventory(policy, json.loads(first.request)["policy"])
    initial = snapshot(left.gradients, first)
    changed = snapshot(right.gradients, second)
    complete(initial, expected)
    complete(changed, expected)
    difference = differences(initial, changed)
    return {"comparison": "gradient tensor bytes", "equal": not difference,
            "left_digest": first.digest, "right_digest": second.digest,
            "left_binding": first.invocation.binding(), "right_binding": second.invocation.binding(),
            "left_tensors": len(initial), "right_tensors": len(changed), "differences": difference}


@dataclass(frozen=True, kw_only=True)
class Input:
    gradients: Path
    log: Path
    call: int


def arguments():
    parser = argparse.ArgumentParser(description="Compare bound pre-AdamW gradient observations bitwise")
    parser.add_argument("--policy", type=Path, required=True,
                        help="Input adapter checkpoint whose tensor identity is consumed by both updates")
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-gradients", type=Path, required=True)
        parser.add_argument(f"--{side}-log", type=Path, required=True)
        parser.add_argument(f"--{side}-call", type=int, required=True)
    values = vars(parser.parse_args())
    sides = tuple(Input(**{field: values[f"{side}_{field}"] for field in ("gradients", "log", "call")})
                  for side in ("left", "right"))
    return *sides, values["policy"]


def main():
    result = compare(*arguments())
    print(canonical(result))
    raise SystemExit(0 if result["equal"] else 1)


if __name__ == "__main__":
    main()
