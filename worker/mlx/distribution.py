import mlx.core as mx

from worker.distribution import Observed, Probe, Snapshot
from worker.mlx.words import words


class Capture:
    def __init__(self, probe, *, store=None):
        if not isinstance(probe, Probe):
            raise ValueError("Native full-vocabulary capture requires a checked probe")
        self.probe = probe
        if store is not None and store.probe != probe:
            raise ValueError("Native capture storage differs from the selected probe")
        self.store = store
        self.selected = frozenset(probe.steps)
        self.snapshots = []
        self.previous = None

    def observe(self, step, distribution):
        if step not in self.selected:
            return
        if distribution.dtype != mx.float32 or distribution.ndim != 2 or distribution.shape[0] != 1:
            raise ValueError("Native capture requires one complete FP32 probability vector")
        snapshot = Snapshot(step=step, probability_bits=words(distribution.reshape(-1)))
        if self.previous is not None:
            previous_step, previous_width = self.previous
            if step <= previous_step:
                raise ValueError("Native capture repeated or reordered a selected step")
            if len(snapshot.probability_bits) != previous_width:
                raise ValueError("Native capture changed vocabulary during the path")
        if self.store is None:
            self.snapshots.append(snapshot)
        else:
            self.store.append(snapshot)
        self.previous = (step, len(snapshot.probability_bits))

    def completed(self):
        if self.store is not None:
            return self.store.completed()
        return Observed(probe=self.probe, snapshots=tuple(self.snapshots))
