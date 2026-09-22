from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import BinaryIO

from worker import core, probe
from worker.distribution import Probe
from worker.probejson import Location, Reader
from worker.probeschema import Cost, Relation, cost
from worker.scoring import Source


@dataclass(frozen=True, kw_only=True)
class Entry:
    location: Location
    step: int
    vocabulary: int


@dataclass(frozen=True, kw_only=True)
class Snapshots:
    stream: BinaryIO
    entries: tuple[Entry, ...]
    source: Source
    words: tuple[int, ...]
    relation: Relation

    def __iter__(self):
        for entry in self.entries:
            self.stream.seek(entry.location.offset)
            encoded = self.stream.read(entry.location.size)
            if hashlib.sha256(encoded).hexdigest() != entry.location.sha256:
                raise ValueError("Probe snapshot bytes changed after indexing")
            selected = probe.snapshot(core.decode(encoded))
            probe.selected_support(selected, self.source, entry.vocabulary, encoded=self.words, relation=self.relation)
            yield selected


@dataclass(frozen=True, kw_only=True)
class Vectors:
    probe: Probe
    vocabulary: int
    snapshots: Snapshots


@dataclass(frozen=True, kw_only=True)
class Indexed:
    digest: str
    byte_count: int
    source: Source
    target: tuple[tuple[str, str], ...]
    vectors: Vectors
    cost: Cost


def inventory(reader):
    value = {}
    for key in reader.members():
        if key != "snapshots":
            value[key], _ = reader.value()
            continue
        entries = []
        for _ in reader.items("[", "]"):
            encoded, location = reader.value()
            selected = probe.snapshot(encoded)
            entries.append(Entry(location=location, step=selected.step, vocabulary=len(selected.probability_bits)))
        value[key] = entries
    return value


def index(stream):
    reader = Reader(stream)
    value = {}
    for key in reader.members():
        if key == "full_vocabulary":
            value[key] = inventory(reader)
        else:
            value[key], _ = reader.value()
    reader.finish()
    selected, target, relation = probe.metadata(value)
    vectors = value["full_vocabulary"]
    planned = probe.vector_plan(vectors, selected)
    entries = tuple(vectors["snapshots"])
    if tuple(entry.step for entry in entries) != planned.steps:
        raise ValueError("Distribution snapshots must cover exactly the prescribed steps")
    if any(entry.vocabulary != vectors["vocabulary"] for entry in entries):
        raise ValueError("Probe vocabulary size differs from its full vectors")
    probe.path_words(value["log_probability_bits"], selected)
    snapshots = Snapshots(stream=stream, entries=entries, source=selected,
                          words=tuple(value["log_probability_bits"]), relation=relation)
    return Indexed(digest=reader.digest.hexdigest(), byte_count=reader.byte_count, source=selected,
                   target=target, vectors=Vectors(probe=planned, vocabulary=vectors["vocabulary"], snapshots=snapshots),
                   cost=cost(value["measurements"], relation))


@contextmanager
def open_probe(path):
    with Path(path).open("rb") as stream:
        yield index(stream)
