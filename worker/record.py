import hashlib
import json
from dataclasses import dataclass

from worker import scalar

FORMAT = "invar-probabilities-v3"
ROLES = ("behavior", "proximal", "reference", "current", "advantage")

@dataclass(frozen=True, kw_only=True)
class Observation:
    sample: str
    dtype: str
    words: tuple[tuple[int, ...], ...]
    active: tuple[bool, ...]


@dataclass(frozen=True, kw_only=True)
class Evaluated(Observation):
    objective: tuple[scalar.Output, ...]
    linearized: tuple[int, ...]


def cotangents(observed, profile, *, total, materialize, check, linearized):
    inputs = tuple(scalar.Inputs(**dict(zip(ROLES, values, strict=True)))
                   for values in zip(*observed.words, strict=True))
    result = scalar.calculate(profile, total, inputs)
    expected = tuple(item.gradient for item in result), tuple(item.reward_gradient for item in result)
    objective, reward = (materialize(value) for value in expected)
    actual = check(expected, (objective, reward))
    captured = tuple(scalar.Output(term=item.term, gradient=full, reward_gradient=only_reward)
                     for item, full, only_reward in zip(result, *actual, strict=True))
    evaluated = Evaluated(sample=observed.sample, dtype=observed.dtype, words=observed.words,
                          active=observed.active, objective=captured, linearized=linearized)
    return evaluated, objective, reward


def loss(observations):
    return scalar.mean32(tuple(value.term for item in observations for value in item.objective))


def document(observations, invocation, request):
    samples = [{"sample": item.sample, "dtype": item.dtype,
                **{role: list(value) for role, value in zip(ROLES, item.words, strict=True)},
                "active": list(item.active), "objective": scalar.document(item.objective),
                "linearized": list(item.linearized)} for item in observations]
    return {"format": FORMAT, "invocation": invocation, "request": request, "samples": samples,
            "scalar_reference": scalar.REFERENCE, "loss": loss(observations)}


def save(path, observations, *, invocation, request):
    encoded = json.dumps(document(observations, invocation, request), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    with path.open("xb") as output:
        output.write(encoded)
    return hashlib.sha256(encoded).hexdigest()
