from types import MappingProxyType

from worker.distribution import Probe
from worker.probepacked import Packed, Snapshots, packed
from worker.vllm.distribution import Position, capture


class Probes:
    def __init__(self, paths, declared):
        if not declared or not set(declared).issubset(paths):
            raise ValueError("Native probes must select owned prescribed requests")
        for key, probe in declared.items():
            if not isinstance(probe, Probe):
                raise ValueError("Native probe declarations require checked immutable steps")
            probe.validate(len(paths[key]))
        self.declared = MappingProxyType(dict(declared))
        self.snapshots = {key: [] for key in declared}
        self.finished = None

    def positions(self, monitor):
        result = []
        for row, value in enumerate(monitor.pending.rows):
            key = value.request
            if key not in self.declared or key not in monitor.sampling_requests:
                continue
            step = len(monitor.tokens[key])
            if step in self.declared[key].steps:
                result.append(Position(row=row, step=step))
        return tuple(result)

    def sample(self, operation, *, logits, monitor):
        if self.finished is not None:
            raise RuntimeError("Native probe observations are already complete")
        selected = self.positions(monitor)
        if not selected:
            return operation()
        requests = tuple(row.request for row in monitor.pending.rows)
        result, observed = capture(operation, shape=tuple(logits.shape), positions=selected)
        for position, snapshot in observed:
            self.snapshots[requests[position.row]].append(packed(snapshot))
        return result

    def completed(self):
        self.finished = MappingProxyType({key: Packed(probe=probe, snapshots=Snapshots(rows=tuple(self.snapshots[key])))
                                          for key, probe in self.declared.items()})

    def observation(self, key):
        if self.finished is None:
            raise RuntimeError("Native mass observations have not completed")
        return self.finished.get(key)
