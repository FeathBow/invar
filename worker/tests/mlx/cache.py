import unittest

try:
    import mlx  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import unittest

import mlx.core as mx

from worker.mlx.cache import BoundedArraysCache, TypedBatchKVCache, TypedKVCache
from worker.mlx import tensors as mlx_tensors

HEADS = 2
WIDTH = 16
TOKENS = 5
RECURRENT_CACHES = 48
DECODE_STEPS = 12000


class CacheTests(unittest.TestCase):
    def test_unread_metadata_cannot_retain_a_graph_per_decode_step(self):
        caches = [BoundedArraysCache.merge([BoundedArraysCache(2), BoundedArraysCache(2)])
                  for _ in range(RECURRENT_CACHES)]
        for cache in caches:
            cache[1] = mx.zeros((2, 1), dtype=mx.float32)
        for _ in range(DECODE_STEPS):
            mx.eval(caches[0].make_mask(1))
            outputs = []
            for cache in caches:
                value = cache[1] + 1
                cache[1] = value
                outputs.append(value)
                cache.advance(1)
            mx.eval(outputs)
        self.assertTrue(all(cache[1].tolist() == [[float(DECODE_STEPS)], [float(DECODE_STEPS)]] for cache in caches))

    def test_empty_join_preserves_dtypes_and_the_incoming_cache(self):
        keys = mx.arange(HEADS * TOKENS * WIDTH).reshape(1, HEADS, TOKENS, WIDTH).astype(mx.bfloat16)
        values = -keys
        for empty_first in (False, True):
            with self.subTest(empty_first=empty_first):
                occupied, empty = TypedBatchKVCache([0]), TypedBatchKVCache([0])
                occupied.update_and_fetch(keys, values)
                left, right = (empty, occupied) if empty_first else (occupied, empty)
                left.extend(right)
                mx.eval(left.state)
                self.assertEqual((left.keys.dtype, left.values.dtype), (keys.dtype, values.dtype))
                if not empty_first:
                    self.assertIsNone(right.keys)
                    self.assertIsNone(right.values)
                continued = copy.deepcopy(left)
                continued.filter([0 if empty_first else 1])
                actual_keys, actual_values = continued.update_and_fetch(keys, values)
                self.assertEqual(continued.offset.item(), TOKENS)
                padding = continued.left_padding.item()
                self.assertTrue(mlx_tensors.equal({"k": actual_keys[:, :, padding:], "v": actual_values[:, :, padding:]},
                                                 {"k": keys, "v": values}))

    def test_merged_empty_cache_retains_its_typed_extend_operation(self):
        merged = TypedKVCache.merge([TypedKVCache(), TypedKVCache()])
        self.assertIs(type(merged), TypedBatchKVCache)

    def test_right_prefill_padding_does_not_advance_recurrent_state(self):
        cache = BoundedArraysCache.merge([BoundedArraysCache(2), BoundedArraysCache(2)])
        cache.prepare(lengths=[3, TOKENS], right_padding=[TOKENS - 3, 0])
        expected = mx.array([[True, True, True, False, False], [True] * TOKENS])
        self.assertTrue(mlx_tensors.equal({"mask": cache.make_mask(TOKENS)}, {"mask": expected}))
        cache.left_padding = mx.array([1, 2])
        expected = mx.array([[False, True, True, False, False], [False, False, True, True, True]])
        self.assertTrue(mlx_tensors.equal({"mask": cache.make_mask(TOKENS)}, {"mask": expected}))


if __name__ == "__main__":
    unittest.main()
