import mlx.core as mx

from worker.mlx.words import words


def tensor(encoded):
    return mx.array(encoded, dtype=mx.uint32).view(mx.float32)
