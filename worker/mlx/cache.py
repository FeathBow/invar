import copy

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_lm.models.qwen3_5 import Model


def empty_tensors(empty, occupied):
    count = empty.offset.shape[0]
    keys, values = occupied.keys, occupied.values
    return (mx.zeros((count, keys.shape[1], 0, keys.shape[3]), dtype=keys.dtype),
            mx.zeros((count, values.shape[1], 0, values.shape[3]), dtype=values.dtype))


class TypedBatchKVCache(BatchKVCache):
    def extend(self, other):
        if self.keys is None and other.keys is not None:
            self.keys, self.values = empty_tensors(self, other)
        incoming = other
        if other.keys is None and self.keys is not None:
            incoming = copy.copy(other)
            incoming.keys, incoming.values = empty_tensors(other, self)
        super().extend(incoming)


class TypedKVCache(KVCache):
    @classmethod
    def merge(cls, caches):
        result = BatchKVCache.merge(caches)
        object.__setattr__(result, "__class__", TypedBatchKVCache)
        return result


class BoundedArraysCache(ArraysCache):
    def advance(self, count):
        super().advance(count)
        dependencies = [value for value in (self.lengths, self.left_padding) if value is not None]
        if dependencies:
            # Every recurrent state consumer also evaluates its metadata. The
            # model reads a shared mask from just one recurrent layer's cache.
            self.cache = [mx.depends(value, dependencies) if value is not None else None for value in self.cache]

    def make_mask(self, length):
        if self.lengths is None:
            return super().make_mask(length)
        positions = mx.arange(length)
        valid = positions < self.lengths[:, None]
        return valid if self.left_padding is None else valid & (positions >= self.left_padding[:, None])


class CachedModel(Model):
    def make_cache(self):
        caches = super().make_cache()
        implementations = {KVCache: TypedKVCache, ArraysCache: BoundedArraysCache}
        if any(type(cache) not in implementations for cache in caches):
            raise TypeError("The pinned native model has an undeclared cache implementation")
        for cache in caches:
            object.__setattr__(cache, "__class__", implementations[type(cache)])
        return caches


def install(model):
    if type(model) is not Model:
        raise TypeError("Native cache correction requires the pinned Qwen3.5 model")
    object.__setattr__(model, "__class__", CachedModel)
