import mlx.core as mx

from worker import record as record
from worker import scalar


def words(value):
    if value.dtype != mx.float32 or value.ndim != 1:
        raise ValueError("Expected a native FP32 probability vector")
    return tuple(value.view(mx.uint32).tolist())


def tensor(encoded):
    return mx.array(encoded, dtype=mx.uint32).view(mx.float32)


def checked(sample, roles, active, *, advantage, count):
    if set(roles) != set(record.ROLES) or count <= 0 or active.dtype != mx.bool_ or active.shape != (count,):
        raise ValueError("Objective mask must describe every admitted response token")
    if any(value.dtype != mx.float32 or value.shape != (count,) for value in roles.values()):
        raise ValueError("Admitted objective inputs must be matching native FP32 response vectors")
    observed = record.Observation(sample=sample, dtype="F32", words=tuple(words(roles[role]) for role in record.ROLES),
                                  active=tuple(active.tolist()))
    if not all(observed.active) or observed.words[record.ROLES.index("advantage")] != (scalar.word(advantage),) * count:
        raise ValueError("Actual objective mask or advantage differs from the checked sample")
    return observed


def checked_cotangents(expected, actual):
    if len(actual) != len(expected):
        raise ValueError("Scalar cotangent role inventory mismatch")
    for value, intended in zip(actual, expected, strict=True):
        if value.dtype != mx.float32 or value.shape != (len(intended),):
            raise ValueError("Actual scalar cotangents must be native FP32 vectors")
    observed = tuple(words(value) for value in actual)
    if observed != expected:
        raise ValueError("Actual scalar cotangents differ from the calculated FP32 words")
    return observed


def cotangents(observed, profile, *, total):
    return record.cotangents(observed, profile, total=total, materialize=tensor, check=checked_cotangents)
