import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import BatchKVCache, KVCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention

from worker.mlx.recurrence import bind

QUERY_TOKENS = 8
SPLIT_QUERIES = 64


def align(keys, values, mask, *, left_padding):
    length = keys.shape[2]
    indices = (mx.arange(length)[None] + left_padding[:, None]) % length
    return (mx.take_along_axis(keys, indices[:, None, :, None], axis=2),
            mx.take_along_axis(values, indices[:, None, :, None], axis=2),
            mx.take_along_axis(mask, indices[:, None, None, :], axis=-1))


def attention(queries, keys, values, *, cache, mask, scale):
    batch, heads, length, _ = queries.shape
    if batch == 1 and length >= SPLIT_QUERIES:
        return scaled_dot_product_attention(queries, keys, values, cache=cache, scale=scale, mask=mask)
    width = keys.shape[-2]
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError("The pinned native attention requires a causal mask")
        mask = mx.arange(width)[None, :] <= mx.arange(width - length, width)[:, None]
    elif mask is None:
        mask = mx.ones((length, width), dtype=mx.bool_)
    mask = mx.broadcast_to(mask, (batch, heads, length, width))
    if isinstance(cache, BatchKVCache):
        keys, values, mask = align(keys, values, mask, left_padding=cache.left_padding)
    elif cache is not None and not isinstance(cache, KVCache):
        raise TypeError("The declared native attention requires an unquantized KV cache")
    outputs = []
    for start in range(0, length, QUERY_TOKENS):
        interval = slice(start, start + QUERY_TOKENS)
        outputs.append(fixed_query_block(queries[:, :, interval], keys, values, mask=mask[:, :, interval], scale=scale))
    return mx.concatenate(outputs, axis=2)


def fixed_query_block(queries, keys, values, *, mask, scale):
    length = queries.shape[2]
    padding = QUERY_TOKENS - length
    if padding:
        queries = mx.concatenate((queries, mx.repeat(queries[:, :, -1:], padding, axis=2)), axis=2)
        mask = mx.concatenate((mask, mx.repeat(mask[:, :, -1:], padding, axis=2)), axis=2)
    return query_block(queries, keys, values, mask=mask, scale=scale)[:, :, :length]


def query_block(queries, keys, values, *, mask, scale):
    batch, heads, length, dimension = queries.shape
    queries = queries.reshape(batch, heads * length, 1, dimension)
    mask = mask.reshape(batch, heads * length, 1, keys.shape[-2])
    result = mx.fast.scaled_dot_product_attention(queries, keys, values, mask=mask, scale=scale)
    return result.reshape(batch, heads, length, values.shape[-1])


class QueryAttention(Qwen3NextAttention):
    __call__ = bind(Qwen3NextAttention.__call__, name="scaled_dot_product_attention", operation=attention)
