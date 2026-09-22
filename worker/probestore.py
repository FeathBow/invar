from dataclasses import dataclass
import hashlib
import os
import struct
from typing import BinaryIO

from worker.distribution import FP32_BYTES, Probe, Snapshot


@dataclass(frozen=True, kw_only=True)
class Entry:
    step: int
    offset: int
    size: int
    sha256: bytes


@dataclass(frozen=True, kw_only=True)
class Snapshots:
    stream: BinaryIO
    entries: tuple[Entry, ...]

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        expected = sum(entry.size for entry in self.entries)
        if self.stream.seek(0, os.SEEK_END) != expected:
            raise ValueError("Probe storage length changed after capture")
        for entry in self.entries:
            self.stream.seek(entry.offset)
            encoded = self.stream.read(entry.size)
            if hashlib.sha256(encoded).digest() != entry.sha256:
                raise ValueError("Probe vector bytes changed after capture")
            words = tuple(value for value, in struct.iter_unpack("<I", encoded))
            yield Snapshot(step=entry.step, probability_bits=words)


@dataclass(frozen=True, kw_only=True)
class Stored:
    probe: Probe
    vocabulary: int
    snapshots: Snapshots


class Store:
    def __init__(self, stream, *, probe):
        if not isinstance(probe, Probe):
            raise ValueError("Probe storage requires an immutable selection")
        if stream.seek(0, os.SEEK_END) != 0:
            raise ValueError("Probe storage requires an empty owned file")
        self.stream = stream
        self.probe = probe
        self.entries = []
        self.vocabulary = None
        self.finished = None

    def append(self, snapshot):
        if self.finished is not None:
            raise RuntimeError("Probe storage is already complete")
        if not isinstance(snapshot, Snapshot):
            raise ValueError("Probe storage requires a checked FP32 snapshot")
        position = len(self.entries)
        if position >= len(self.probe.steps) or snapshot.step != self.probe.steps[position]:
            raise ValueError("Probe storage requires exactly the selected steps in order")
        width = len(snapshot.probability_bits)
        if self.vocabulary is not None and width != self.vocabulary:
            raise ValueError("Probe storage changed vocabulary during capture")
        encoded = struct.pack(f"<{width}I", *snapshot.probability_bits)
        offset = self.stream.seek(0, os.SEEK_END)
        if offset != position * width * FP32_BYTES:
            raise ValueError("Probe storage length differs from completed writes")
        if self.stream.write(encoded) != len(encoded):
            raise OSError("Incomplete probe vector write")
        self.entries.append(Entry(step=snapshot.step, offset=offset, size=len(encoded),
                                  sha256=hashlib.sha256(encoded).digest()))
        self.vocabulary = width

    def completed(self):
        if len(self.entries) != len(self.probe.steps):
            raise ValueError("Probe storage must cover exactly the complete selected steps")
        self.stream.flush()
        if self.stream.seek(0, os.SEEK_END) != len(self.entries) * self.vocabulary * FP32_BYTES:
            raise ValueError("Probe storage size differs from its complete inventory")
        if self.finished is None:
            self.finished = Stored(probe=self.probe, vocabulary=self.vocabulary,
                                   snapshots=Snapshots(stream=self.stream, entries=tuple(self.entries)))
        return self.finished
