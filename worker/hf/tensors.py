import hashlib
import json
import struct

import torch

FINGERPRINT_FORMAT = "invar-value-fingerprint-v1"


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(tensor_bytes(value).numpy().tobytes())
    return result.hexdigest()


def fingerprint(value):
    encoded = json.dumps([FINGERPRINT_FORMAT, content(value)], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def content(value):
    if isinstance(value, torch.Tensor):
        return ["tensor", digest({"": value})]
    if type(value) is dict:
        items = [(content(key), content(item)) for key, item in value.items()]
        return ["mapping", sorted(items, key=lambda pair: json.dumps(pair[0]))]
    if type(value) in (list, tuple):
        return [type(value).__name__, [content(item) for item in value]]
    if type(value) is float:
        return ["float", struct.pack("!d", value).hex()]
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    raise TypeError(f"Unsupported checkpoint fingerprint value: {type(value).__name__}")


def assert_equal(first, second):
    if type(first) is not type(second):
        raise RuntimeError("Checkpoint type mismatch")
    if isinstance(first, torch.Tensor):
        if first.dtype != second.dtype or first.shape != second.shape:
            raise RuntimeError("Checkpoint tensor metadata mismatch")
        if not torch.equal(tensor_bytes(first), tensor_bytes(second)):
            raise RuntimeError("Checkpoint tensor mismatch")
        return
    if isinstance(first, dict):
        assert_mapping(first, second)
        return
    if isinstance(first, (list, tuple)):
        assert_sequence(first, second)
        return
    if first != second:
        raise RuntimeError("Checkpoint value mismatch")


def assert_mapping(first, second):
    if first.keys() != second.keys():
        raise RuntimeError("Checkpoint key mismatch")
    for key in first:
        assert_equal(first[key], second[key])


def assert_sequence(first, second):
    if len(first) != len(second):
        raise RuntimeError("Checkpoint length mismatch")
    for left, right in zip(first, second):
        assert_equal(left, right)
