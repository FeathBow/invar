import hashlib
import json
import struct
import sys

from worker import core

CHUNK_BYTES = 64 * 1024


def typename(value):
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def write(target, encoded):
    content = memoryview(encoded)
    offset = 0
    while offset < len(content):
        count = target.write(content[offset:])
        if not count:
            raise OSError("Native codec output stopped before the complete response")
        offset += count


class Session:
    def __init__(self, *, load, describe, view):
        self.load = load
        self.describe = describe
        self.view = view
        self.tensors = []
        self.views = {}

    def encode(self, value):
        tensor = self.describe(value)
        if tensor is not None:
            index = len(self.tensors)
            self.tensors.append(value)
            return {"kind": "tensor", "type": typename(value), "index": index, **tensor}
        if isinstance(value, dict):
            return {"kind": "mapping", "type": typename(value),
                    "items": [[self.encode(key), self.encode(item)] for key, item in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"kind": "list" if isinstance(value, list) else "tuple", "type": typename(value),
                    "items": [self.encode(item) for item in value]}
        if type(value) is bool:
            return {"kind": "boolean", "value": value}
        if type(value) is int:
            return {"kind": "integer", "hex": format(value, "x")}
        if type(value) is float:
            return {"kind": "float", "bits": struct.unpack("!Q", struct.pack("!d", value))[0]}
        if type(value) is str:
            return {"kind": "string", "value": value}
        if value is None:
            return {"kind": "none"}
        return {"kind": "unsupported", "type": typename(value)}

    def decode(self, path):
        with open(path, "rb") as source:
            encoded = source.read()
        return {"format": "invar-native-checkpoint/v1",
                "source_sha256": hashlib.sha256(encoded).hexdigest(),
                "byte_order": "little", "value": self.encode(self.load(encoded))}

    def tensor(self, request, target):
        index, offset, count = (request[key] for key in ("index", "offset", "count"))
        if any(type(value) is not int or value < 0 for value in (index, offset, count)):
            raise ValueError("Expected nonnegative tensor slice coordinates")
        if index not in self.views:
            self.views[index] = self.view(self.tensors[index])
        encoded = self.views[index]
        if offset + count > len(encoded):
            raise ValueError("Native tensor slice exceeds its data")
        for start in range(offset, offset + count, CHUNK_BYTES):
            write(target, encoded[start:min(start + CHUNK_BYTES, offset + count)])
        target.flush()

    def handle(self, request, target):
        if request.get("codec") == "decode" and set(request) == {"codec", "path"}:
            value = self.decode(request["path"])
            write(target, json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n")
            target.flush()
        elif request.get("codec") == "tensor" and set(request) == {"codec", "index", "offset", "count"}:
            self.tensor(request, target)
        elif request == {"codec": "release"}:
            self.tensors = []
            self.views = {}
            write(target, b'{"released":true}\n')
            target.flush()
        else:
            raise ValueError("Unknown native checkpoint codec request")


def run(session):
    for line in sys.stdin.buffer:
        session.handle(core.decode(line), sys.stdout.buffer)
