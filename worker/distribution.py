from dataclasses import dataclass
import math

from worker.float32 import number

FP32_BYTES = 4
PROBE_FORMAT = "invar-cached-distribution-probe-v1"


@dataclass(frozen=True, kw_only=True)
class Probe:
    steps: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("A full-vocabulary probe requires an immutable nonempty step selection")
        if any(type(step) is not int or step < 0 for step in self.steps):
            raise ValueError("Probe steps must be nonnegative integers")
        if any(left >= right for left, right in zip(self.steps, self.steps[1:])):
            raise ValueError("Probe steps must be strictly increasing without duplicates")

    def validate(self, length):
        if type(length) is not int or length <= 0 or self.steps[-1] >= length:
            raise ValueError("Probe steps must lie inside the complete prescribed response")


@dataclass(frozen=True, kw_only=True)
class Snapshot:
    step: int
    probability_bits: tuple[int, ...]

    def __post_init__(self):
        if type(self.step) is not int or self.step < 0:
            raise ValueError("A distribution snapshot requires a nonnegative response step")
        if not isinstance(self.probability_bits, tuple) or not self.probability_bits:
            raise ValueError("A distribution snapshot requires every vocabulary word")
        values = tuple(number(word) for word in self.probability_bits)
        if any(value < 0 or value > 1 for value in values) or math.fsum(values) <= 0:
            raise ValueError("A distribution snapshot requires nonnegative probability masses with positive total")


@dataclass(frozen=True, kw_only=True)
class Observed:
    probe: Probe
    snapshots: tuple[Snapshot, ...]

    def __post_init__(self):
        if not isinstance(self.probe, Probe) or not isinstance(self.snapshots, tuple):
            raise ValueError("A distribution observation requires a probe and immutable snapshots")
        if any(not isinstance(value, Snapshot) for value in self.snapshots):
            raise ValueError("A distribution observation contains an invalid snapshot")
        if tuple(value.step for value in self.snapshots) != self.probe.steps:
            raise ValueError("Distribution snapshots must cover exactly the prescribed steps")
        if len({len(value.probability_bits) for value in self.snapshots}) != 1:
            raise ValueError("Distribution snapshots must share a complete vocabulary")

    @property
    def vocabulary(self):
        return len(self.snapshots[0].probability_bits)
