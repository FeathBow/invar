import hashlib
import struct
from dataclasses import asdict, dataclass

from worker.cohort import Distillation, Distilled, Shared, decode as cohort
from worker.invocation import Invocation, decode as invocation

@dataclass(frozen=True, kw_only=True)
class Call:
    invocation: Invocation
    load: Invocation
    request: Shared


def decode(value):
    if not isinstance(value, dict) or set(value) != {"invocation", "request", "load"}:
        raise ValueError("Expected a bound update invocation and its numerical request")
    bound = invocation(value["invocation"])
    loading = invocation(value["load"])
    if loading.binding() != bound.binding():
        raise ValueError("Learner load must correlate with the update binding")
    return Call(invocation=bound, load=loading, request=cohort(value["request"]))


def snapshot(path):
    encoded = path.read_bytes()
    return hashlib.sha256(encoded).hexdigest(), encoded


def observation(trajectory, item):
    if trajectory.request.sample != item.sample or trajectory.request.group != item.group:
        raise ValueError("Actual trajectory differs from the requested sample")
    return {**{name: value for name, value in asdict(trajectory.request).items() if name != "group"}, "tokens": trajectory.tokens[0].tolist(),
            "prompt_length": trajectory.prompt_length,
            "version": item.version, "behavior_policy": item.behavior_policy,
            "behavior_bits": [struct.unpack("!I", struct.pack("!f", value))[0] for value in trajectory.behavior.tolist()],
            "reference_bits": list(item.reference_bits),
            "text": trajectory.text, "truncated": trajectory.truncated, **signal(item)}


def signal(item):
    if isinstance(item, Distilled):
        return {"teacher_bits": list(item.teacher_bits), "teacher_call": asdict(item.teacher_call)}
    return {"group": item.group, "reward": item.reward, "advantage_bits": item.advantage_bits}


def consumed(request, *, trajectories, loaded):
    values = {item.sample: item for item in request.samples}
    if len(trajectories) != len(values) or {item.request.sample for item in trajectories} != set(values):
        raise ValueError("Actual trajectories must match the requested samples")
    samples = [observation(item, values[item.request.sample]) for item in trajectories]
    return {"specification": request.specification, **loaded,
            "behavior_model": asdict(request.behavior_model), "schedule": asdict(request.schedule), "samples": samples,
            "reference_source": request.reference_source,
            "order": list(request.order), "steps": [list(batch) for batch in request.steps],
            "epsilon": request.epsilon, "penalty": request.penalty, **declared(request)}


def declared(request):
    if isinstance(request, Distillation):
        return {"teacher": asdict(request.teacher)}
    return {"delta": request.delta}
