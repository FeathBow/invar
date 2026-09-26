import mlx.core as mx


def words(value):
    if value.dtype != mx.float32 or value.ndim != 1:
        raise ValueError("Expected a native FP32 probability vector")
    return tuple(value.view(mx.uint32).tolist())

