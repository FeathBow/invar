import unittest

try:
    import mlx  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

import mlx.core as mx

from mlx_lm.models.base import scaled_dot_product_attention

from worker.mlx.attention import SPLIT_QUERIES, attention, query_block
from worker.mlx.cache import TypedBatchKVCache
from worker.mlx import tensors as mlx_tensors

QUERY_HEADS = 24
KEY_HEADS = 4
HEAD_WIDTH = 256
VALID_TOKENS = 37
PADDED_TOKENS = 48
STOCK_TOKENS = 67
LEFT_PADDING = 79
SCALE = HEAD_WIDTH ** -0.5


def values(heads, length):
    return mx.random.normal((1, heads, length, HEAD_WIDTH)).astype(mx.bfloat16)


class AttentionTests(unittest.TestCase):
    def test_query_blocks_and_mixed_prefill_preserve_all_valid_outputs(self):
        mx.random.seed(79)
        queries = values(QUERY_HEADS, PADDED_TOKENS)
        keys, responses = (values(KEY_HEADS, PADDED_TOKENS) for _ in range(2))
        causal = mx.arange(VALID_TOKENS)[None, :] <= mx.arange(VALID_TOKENS)[:, None]
        original = attention(queries[:, :, :VALID_TOKENS], keys[:, :, :VALID_TOKENS], responses[:, :, :VALID_TOKENS],
                             cache=None, mask="causal", scale=SCALE)
        full_mask = mx.broadcast_to(causal, (1, QUERY_HEADS, VALID_TOKENS, VALID_TOKENS))
        unblocked = query_block(queries[:, :, :VALID_TOKENS], keys[:, :, :VALID_TOKENS], responses[:, :, :VALID_TOKENS],
                                mask=full_mask, scale=SCALE)
        self.assertTrue(mlx_tensors.equal({"output": original}, {"output": unblocked}))
        mixed = tuple(mx.concatenate((value, mx.flip(value, axis=2))) for value in (queries, keys, responses))
        positions = mx.arange(PADDED_TOKENS)
        causal = positions[None, :] <= positions[:, None]
        valid = positions[None, None, None, :] < mx.array([VALID_TOKENS, PADDED_TOKENS])[:, None, None, None]
        actual = attention(*mixed, cache=TypedBatchKVCache([0, 0]), mask=causal & valid, scale=SCALE)
        self.assertTrue(mlx_tensors.equal({"output": actual[:1, :, :VALID_TOKENS]}, {"output": original}))

    def test_one_sequence_of_stock_queries_uses_the_library_attention(self):
        mx.random.seed(89)
        queries = values(QUERY_HEADS, STOCK_TOKENS)
        keys, responses = (values(KEY_HEADS, STOCK_TOKENS) for _ in range(2))
        actual = attention(queries, keys, responses, cache=None, mask="causal", scale=SCALE)
        stock = scaled_dot_product_attention(queries, keys, responses, cache=None, scale=SCALE, mask="causal")
        self.assertGreaterEqual(STOCK_TOKENS, SPLIT_QUERIES)
        self.assertTrue(mlx_tensors.equal({"output": actual}, {"output": stock}))

    def test_masked_left_padding_preserves_the_logical_decode(self):
        mx.random.seed(83)
        queries = values(QUERY_HEADS, 1)
        keys, responses = (values(KEY_HEADS, VALID_TOKENS) for _ in range(2))
        original = attention(queries, keys, responses, cache=None, mask=None, scale=SCALE)
        padding = ((0, 0), (0, 0), (LEFT_PADDING, 0), (0, 0))
        keys, responses = (mx.pad(value, padding) for value in (keys, responses))
        mask = mx.arange(LEFT_PADDING + VALID_TOKENS) >= LEFT_PADDING
        actual = attention(queries, keys, responses, cache=TypedBatchKVCache([LEFT_PADDING]), mask=mask, scale=SCALE)
        self.assertTrue(mlx_tensors.equal({"output": actual}, {"output": original}))


if __name__ == "__main__":
    unittest.main()
