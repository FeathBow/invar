from dataclasses import asdict, dataclass
from enum import Enum
import math

from worker.cohort import fields

MLX_ENGINE = "mlx_lm.generate.BatchGenerator"
VLLM_ENGINE = "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
NATIVE_CACHE = "target-owned native request caches; no source cache input"
NATIVE_PATH = "replace sampled ids after native probability calculation"
MASS_CAPTURE = "torch.Tensor.softmax.output/F32/v1"
LOG_CAPTURE = "torch.Tensor.log_softmax.output/F32/v1"
SEPARATE_RELATION = "separately rounded log-softmax and softmax/v1"
WORKER_SCOPE = "native worker permit-to-completion; loading and serialization excluded/v1"
CLIENT_SCOPE = "client wait_for_completion/v1"


class Relation(Enum):
    LOG_OF_MASS = "log_of_represented_mass"
    SEPARATE_LOG_SOFTMAX = "separately_rounded_log_softmax"


def native_relation():
    return {"mass_capture": MASS_CAPTURE, "reported_log": LOG_CAPTURE, "relation": SEPARATE_RELATION}


def execution(value, selected):
    if not isinstance(value, dict):
        raise ValueError("Probe execution must be an object")
    if value.get("engine") == MLX_ENGINE:
        mlx_execution(value, selected)
        return Relation.LOG_OF_MASS
    if value.get("engine") == VLLM_ENGINE:
        native_execution(value, selected)
        return Relation.SEPARATE_LOG_SOFTMAX
    raise ValueError("Unsupported probe execution engine")


def mlx_execution(value, selected):
    fields(value, "engine sampling cache_origin unused_native_lookahead_draws truncated")
    if value["cache_origin"] != "fresh native caches; no reference cache input":
        raise ValueError("Probe does not describe fresh native MLX caches")
    if type(value["unused_native_lookahead_draws"]) is not int or value["unused_native_lookahead_draws"] != 1:
        raise ValueError("Probe unused lookahead differs")
    if value["truncated"] is not selected.truncated:
        raise ValueError("Probe stopping boundary differs")
    sampling = fields(value["sampling"], "batch_size prefill_step")
    if any(type(item) is not int or item <= 0 for item in sampling.values()):
        raise ValueError("Probe native sampling settings must be positive integers")


def native_execution(value, selected):
    fields(value, "engine cache_origin path_control native_sample_rows ignored_prefill_rows truncated distribution")
    if value["cache_origin"] != NATIVE_CACHE or value["path_control"] != NATIVE_PATH:
        raise ValueError("Probe does not describe owned native prescribed caches")
    total, ignored = value["native_sample_rows"], value["ignored_prefill_rows"]
    if type(total) is not int or type(ignored) is not int or ignored < 0 or total - ignored != len(selected.path.response):
        raise ValueError("Probe native row accounting differs from its complete response")
    if value["truncated"] is not selected.truncated:
        raise ValueError("Probe stopping boundary differs")
    if fields(value["distribution"], "mass_capture reported_log relation") != native_relation():
        raise ValueError("Probe log/mass measurement relation is unsupported")


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


@dataclass(frozen=True, kw_only=True)
class Cost:
    seconds: float
    peak_active: int | None = None
    workers: tuple[WorkerCost, ...] = ()

    def describe(self):
        if self.workers:
            return {"scoring_seconds": self.seconds, "seconds_scope": CLIENT_SCOPE,
                    "workers": [asdict(value) for value in self.workers],
                    "memory_scope": "separate worker CUDA allocation peaks; host objects and serialization excluded"}
        return {"scoring_seconds": self.seconds, "peak_active_bytes": self.peak_active,
                "memory_scope": "MLX active allocator peak for scoring; Python objects and JSON encoding excluded"}


def cost(values, relation):
    if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
        raise ValueError("Probe must retain its actual measurement records")
    scores = [value for value in values if value.get("stage") == "cross_score"]
    if len(scores) != 1:
        raise ValueError("Probe requires exactly one actual cached-scoring measurement")
    if relation is Relation.LOG_OF_MASS:
        return mlx_cost(scores[0])
    if relation is Relation.SEPARATE_LOG_SOFTMAX:
        value = fields(scores[0], "stage seconds seconds_scope allocator workers")
        if value["seconds_scope"] != CLIENT_SCOPE or value["allocator"] != "torch.cuda":
            raise ValueError("Probe cost must identify native worker allocations and client timing")
        return Cost(seconds=duration(value["seconds"]), workers=workers(value["workers"]))
    raise ValueError("Unknown probe measurement relation")


def mlx_cost(value):
    measured = fields(value, "stage seconds allocator peak_active cache_end")
    seconds = duration(measured["seconds"])
    if measured["allocator"] != "mlx" or type(measured["peak_active"]) is not int or measured["peak_active"] <= 0:
        raise ValueError("Probe requires the observed MLX peak active allocation")
    if type(measured["cache_end"]) is not int or measured["cache_end"] < 0:
        raise ValueError("Probe cache measurement must be nonnegative")
    return Cost(seconds=seconds, peak_active=measured["peak_active"])
