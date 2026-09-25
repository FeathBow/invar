import hashlib
from dataclasses import asdict, dataclass

from worker.cohort import Cohort, decode as cohort
from worker.invocation import Invocation, decode as invocation

@dataclass(frozen=True, kw_only=True)
class Call:
    invocation: Invocation
    load: Invocation
    request: Cohort


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


def observation(trajectory, reward, *, advantage, reference):
    from worker.advantage import word

    if trajectory.request.sample != reward.sample or trajectory.request.group != reward.group:
        raise ValueError("Actual reward and trajectory identities disagree")
    return {**asdict(trajectory.request), "tokens": trajectory.tokens[0].tolist(),
            "prompt_length": trajectory.prompt_length,
            "behavior_bits": [word(value) for value in trajectory.behavior.tolist()],
            "reference_bits": list(reference),
            "text": trajectory.text, "truncated": trajectory.truncated, "reward": reward.value,
            "advantage_bits": word(advantage)}


def consumed(request, *, batch, rewards, loaded):
    if len(rewards) != len(batch.samples) or len({item.sample for item in rewards}) != len(rewards):
        raise ValueError("Actual rewards must match the prepared samples")
    values = {item.sample: item for item in rewards}
    scores = {item.sample: item.reference_bits for item in request.samples}
    samples = [observation(item.trajectory, values[item.trajectory.request.sample], advantage=item.advantage,
                           reference=scores[item.trajectory.request.sample]) for item in batch.samples]
    return {"specification": request.specification, **loaded,
            "behavior_model": asdict(request.behavior_model), "samples": samples,
            "order": batch.order, "epsilon": batch.profile.epsilon,
            "penalty": batch.profile.penalty, "delta": request.delta}
