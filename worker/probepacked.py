from dataclasses import dataclass
import struct

from worker.distribution import FP32_BYTES, Probe, Snapshot


@dataclass(frozen=True, kw_only=True)
class Row:
    step: int
    payload: bytes

    def decoded(self):
        if type(self.payload) is not bytes or not self.payload or len(self.payload) % FP32_BYTES:
            raise ValueError("Packed probe rows require complete immutable FP32 words")
        words = tuple(word for word, in struct.iter_unpack("<I", self.payload))
        return Snapshot(step=self.step, probability_bits=words)


def packed(snapshot):
    if not isinstance(snapshot, Snapshot):
        raise ValueError("Probe packing requires a checked snapshot")
    width = len(snapshot.probability_bits)
    return Row(step=snapshot.step, payload=struct.pack(f"<{width}I", *snapshot.probability_bits))


@dataclass(frozen=True, kw_only=True)
class Snapshots:
    rows: tuple[Row, ...]

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for row in self.rows:
            yield row.decoded()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(row.decoded() for row in self.rows[index])
        return self.rows[index].decoded()


@dataclass(frozen=True, kw_only=True)
class Packed:
    probe: Probe
    snapshots: Snapshots

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.probe, Probe) or not isinstance(self.snapshots, Snapshots):
            raise ValueError("Packed observations require an immutable probe and snapshot inventory")
        Probe(steps=self.probe.steps)
        rows = self.snapshots.rows
        if type(rows) is not tuple or any(not isinstance(row, Row) for row in rows):
            raise ValueError("Packed observations require immutable encoded rows")
        if tuple(row.step for row in rows) != self.probe.steps:
            raise ValueError("Packed snapshots must cover exactly the prescribed steps")
        widths = set()
        for row in rows:
            widths.add(len(row.decoded().probability_bits))
        if len(widths) != 1:
            raise ValueError("Packed snapshots must share a complete vocabulary")

    @property
    def vocabulary(self):
        return len(self.snapshots.rows[0].payload) // FP32_BYTES
