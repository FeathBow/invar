from dataclasses import dataclass
import math

from worker.cohort import fields

VLLM_ENGINE = "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
NATIVE_CACHE = "target-owned native request caches; no source cache input"
NATIVE_PATH = "replace sampled ids after native probability calculation"
MASS_CAPTURE = "torch.Tensor.softmax.output/F32/v1"
LOG_CAPTURE = "torch.Tensor.log_softmax.output/F32/v1"
SEPARATE_RELATION = "separately rounded log-softmax and softmax/v1"
WORKER_SCOPE = "native worker permit-to-completion; loading and serialization excluded/v1"
CLIENT_SCOPE = "client wait_for_completion/v1"


def native_relation():
    return {"mass_capture": MASS_CAPTURE, "reported_log": LOG_CAPTURE, "relation": SEPARATE_RELATION}


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Probe timing must be a finite nonnegative duration")
    return value


@dataclass(frozen=True, kw_only=True)
class WorkerCost:
    host: str
    pid: int
    device: str
    seconds: float
    peak_allocated: int
    peak_reserved: int
    scope: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in (self.host, self.device)):
            raise ValueError("Probe resources require actual worker host/device identities")
        if type(self.pid) is not int or self.pid <= 0 or self.scope != WORKER_SCOPE:
            raise ValueError("Probe resources require a worker process and declared interval")
        duration(self.seconds)
        if any(type(value) is not int for value in (self.peak_allocated, self.peak_reserved)):
            raise ValueError("Probe resource byte counts must be integers")
        if not 0 < self.peak_allocated <= self.peak_reserved:
            raise ValueError("Probe worker allocation peaks are inconsistent")


def workers(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Probe resources require nonempty worker observations")
    records = tuple(WorkerCost(**fields(value, "host pid device seconds peak_allocated peak_reserved scope")) for value in values)
    identities = {(value.host, value.pid, value.device) for value in records}
    if len(identities) != len(records):
        raise ValueError("Probe resource worker/device identities are repeated")
    return records
