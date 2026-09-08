import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from compare import canonical, paired, unique

ROLES = ("behavior", "proximal", "reference", "current", "advantage")
WORD_LIMIT = 1 << 32


def exact(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Unexpected probability observation fields")


def valid_number(number, *, probability):
    return math.isfinite(number) and (not probability or number <= 0)


def vector(value, count, *, probability):
    if not isinstance(value, list) or len(value) != count or not count:
        raise ValueError("Probability vector token count mismatch")
    if any(type(word) is not int or not 0 <= word < WORD_LIMIT for word in value):
        raise ValueError("Expected unsigned FP32 words")
    numbers = struct.unpack(f"<{count}f", struct.pack(f"<{count}I", *value))
    if any(not valid_number(number, probability=probability) for number in numbers):
        raise ValueError("Invalid probability or advantage floating-point words")


def sample(value, original):
    exact(value, ("sample", "dtype", "active", *ROLES))
    if value["dtype"] != "F32":
        raise ValueError("Update probability observations must use FP32")
    count = len(original["behavior_bits"])
    for role in ROLES:
        vector(value[role], count, probability=role != "advantage")
    if value["behavior"] != original["behavior_bits"]:
        raise ValueError("Behavior probability words differ from consumed input")
    active = value["active"]
    if not isinstance(active, list) or len(active) != count or any(item is not True for item in active):
        raise ValueError("Update probability active mask mismatch")


def validate(value, expected):
    exact(value, ("format", "invocation", "request", "samples"))
    if value["format"] != "invar-probabilities-v1":
        raise ValueError("Unknown probability observation format")
    intended = {"binding": expected.invocation.binding(), "program": expected.invocation.program}
    if canonical(value["invocation"]) != canonical(intended):
        raise ValueError("Probability observation invocation mismatch")
    if canonical(value["request"]) != expected.request:
        raise ValueError("Probability observation request mismatch")
    return samples(value["samples"], value["request"])


def samples(observed, request):
    if not isinstance(observed, list) or any(not isinstance(item, dict) for item in observed):
        raise ValueError("Expected probability sample objects")
    if [item.get("sample") for item in observed] != request["order"]:
        raise ValueError("Probability samples differ from logical order")
    originals = {item["sample"]: item for item in request["samples"]}
    for item in observed:
        sample(item, originals[item["sample"]])
    return observed


def snapshot(path, expected):
    encoded = path.read_bytes()
    digest = json.loads(expected.result)["probabilities"]
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("Probability file differs from its reported digest")
    return validate(json.loads(encoded, object_pairs_hook=unique), expected)


@dataclass(frozen=True, kw_only=True)
class Input:
    probabilities: Path
    log: Path
    call: int


def compare(left, right):
    first, second = paired(left, right)
    initial, changed = snapshot(left.probabilities, first), snapshot(right.probabilities, second)
    differences = [{"sample": old["sample"], "fields": fields}
                   for old, new in zip(initial, changed, strict=True)
                   if (fields := [role for role in ROLES if old[role] != new[role]])]
    return {"comparison": "pre-AdamW objective input words", "equal": not differences,
            "left_binding": first.invocation.binding(), "right_binding": second.invocation.binding(),
            "samples": len(initial), "differences": differences}


def arguments():
    parser = argparse.ArgumentParser(description="Compare bound objective probability words bitwise")
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-probabilities", type=Path, required=True)
        parser.add_argument(f"--{side}-log", type=Path, required=True)
        parser.add_argument(f"--{side}-call", type=int, required=True)
    values = vars(parser.parse_args())
    return tuple(Input(**{field: values[f"{side}_{field}"] for field in ("probabilities", "log", "call")})
                 for side in ("left", "right"))


def main():
    result = compare(*arguments())
    print(canonical(result))
    raise SystemExit(0 if result["equal"] else 1)


if __name__ == "__main__":
    main()
