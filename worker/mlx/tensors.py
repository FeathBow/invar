import hashlib
import json

import mlx.core as mx
import numpy as np

POLICY_METADATA = {"invar_policy": "mlx-f32/v1"}


def describe(value):
    if isinstance(value, mx.array):
        array = np.asarray(value)
        return {"dtype": str(value.dtype), "shape": list(value.shape), "size": value.nbytes,
                "layout": "mlx.row-major" if array.flags.c_contiguous else "mlx.strided"}
    return None


def view(value):
    encoded = np.asarray(value.reshape(-1).view(mx.uint8))
    return memoryview(np.ascontiguousarray(encoded)).cast("B")


def digest(state):
    result = hashlib.sha256()
    if not state:
        raise ValueError("Expected a nonempty native MLX policy")
    for name, value in sorted(state.items()):
        if value.dtype != mx.float32 or not np.isfinite(np.asarray(value)).all():
            raise ValueError("Expected finite FP32 native MLX policy tensors")
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(view(value))
    return result.hexdigest()


def equal(left, right):
    return left.keys() == right.keys() and all(
        left[name].dtype == right[name].dtype and left[name].shape == right[name].shape
        and view(left[name]) == view(right[name]) for name in left)


def policy(path, expected):
    values, metadata = mx.load(path, format="safetensors", return_metadata=True)
    if metadata != POLICY_METADATA or digest(values) != expected:
        raise ValueError("Native MLX policy differs from its declared identity or representation")
    return values


def save_policy(path, values):
    identity = digest(values)
    with path.open("xb") as target:
        mx.save_safetensors(target, values, metadata=POLICY_METADATA)
    return identity
