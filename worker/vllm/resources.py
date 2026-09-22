from dataclasses import asdict
import os
import socket
import time

from worker.probeschema import CLIENT_SCOPE, WORKER_SCOPE, WorkerCost, workers


class Interval:
    def __init__(self, cuda, *, device, clock=time.perf_counter):
        self.cuda, self.device, self.clock = cuda, device, clock
        self.started = None
        self.finished = False

    def start(self):
        if self.started is not None:
            raise RuntimeError("Native probe resource interval already started")
        self.cuda.synchronize(self.device)
        self.cuda.reset_peak_memory_stats(self.device)
        self.started = self.clock()

    def finish(self):
        if self.started is None or self.finished:
            raise RuntimeError("Native probe resource interval is absent or already finished")
        self.cuda.synchronize(self.device)
        observed = WorkerCost(host=socket.gethostname(), pid=os.getpid(), device=str(self.device),
                              seconds=self.clock() - self.started,
                              peak_allocated=self.cuda.max_memory_allocated(self.device),
                              peak_reserved=self.cuda.max_memory_reserved(self.device), scope=WORKER_SCOPE)
        self.finished = True
        return asdict(observed)


class Scoring:
    def __init__(self, measured, *, emit, probing, clock=time.perf_counter):
        self.measured, self.emit, self.probing, self.clock = measured, emit, probing, clock
        self.seconds = None

    def measure(self, stage, operation):
        if stage != "cross_score" or not self.probing:
            return self.measured(stage, operation)
        if self.seconds is not None:
            raise RuntimeError("Native probe client interval already recorded")
        started = self.clock()
        result = operation()
        self.seconds = self.clock() - started
        return result

    def completed(self, observations):
        if not self.probing:
            return
        if self.seconds is None:
            raise RuntimeError("Native probe has no completed client interval")
        resources = workers([value["probe_resources"] for value in observations])
        self.emit("cross_score", {"seconds": self.seconds, "seconds_scope": CLIENT_SCOPE,
                                  "allocator": "torch.cuda", "workers": [asdict(value) for value in resources]})
