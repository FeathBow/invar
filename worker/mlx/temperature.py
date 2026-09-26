import mlx.core as mx


def tempered(logprobs, temperature):
    weights = mx.softmax(logprobs / temperature, axis=-1, precise=True)
    total = mx.sum(weights, axis=-1, keepdims=True)
    return weights / total, weights, total
