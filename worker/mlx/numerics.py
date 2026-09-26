from dataclasses import dataclass
from importlib.metadata import version

import mlx.nn as nn
from mlx_lm.models.qwen3_next import Qwen3NextAttention
from mlx_lm.tuner.lora import LoRALinear

from worker.mlx import cache as mlx_cache
from worker.mlx.attention import QUERY_TOKENS, SPLIT_QUERIES, QueryAttention
from worker.mlx.projection import BITS, GROUP_SIZE, MINIMUM_COLUMNS, MODE, PHYSICAL_ROWS, SPLIT_ROWS, ColumnLoRALinear, RowLinear
from worker.mlx import recurrence as mlx_recurrence


def qualified(implementation):
    return implementation.__module__ + "." + implementation.__qualname__


def inventory(model, family):
    return {name: module for name, module in model.named_modules() if isinstance(module, family)}


@dataclass(frozen=True, kw_only=True)
class Profile:
    name: str
    linear: type
    lora: type
    attention: type

    def families(self):
        return ((nn.QuantizedLinear, self.linear), (LoRALinear, self.lora), (Qwen3NextAttention, self.attention))

    def install(self, model):
        replacements = tuple((inventory(model, family), family, selected) for family, selected in self.families())
        for modules, family, _ in replacements:
            if not modules or any(type(module) is not family for module in modules.values()):
                raise TypeError("Native numerical installation requires the pinned module inventory")
        for module in replacements[0][0].values():
            if (module.bits, module.group_size, module.mode) != (BITS, GROUP_SIZE, MODE):
                raise ValueError("Native numerical installation requires affine4/group64 projections")
        for modules, _, selected in replacements:
            for module in modules.values():
                object.__setattr__(module, "__class__", selected)
        mlx_cache.install(model)
        mlx_recurrence.install(model)

    def observe(self, model):
        if type(model) is not mlx_cache.CachedModel:
            raise TypeError("Native numerical materialization differs from its declared cache model")
        mlx_recurrence.verify(model)
        observed = {}
        for family, selected in self.families():
            modules = inventory(model, family)
            if not modules or any(type(module) is not selected for module in modules.values()):
                raise TypeError("Actual native arithmetic differs from the selected numerical profile")
            if family is nn.QuantizedLinear and any((module.bits, module.group_size, module.mode) != (BITS, GROUP_SIZE, MODE)
                                                   for module in modules.values()):
                raise ValueError("Actual native projections differ from the declared affine4/group64 representation")
            observed[qualified(selected)] = sorted(modules)
        caches = model.make_cache()
        expected = [mlx_cache.BoundedArraysCache if layer.is_linear else mlx_cache.TypedKVCache for layer in model.layers]
        if [type(cache) for cache in caches] != expected:
            raise TypeError("Actual native cache inventory differs from its declared layer roles")
        return {"format": "invar-mlx-numerics/v3", "name": self.name, "modules": observed,
                "projection_rows": PHYSICAL_ROWS if self.linear is RowLinear else None,
                "projection_row_padding": "zero rows to the fixed block; discard padded outputs" if self.linear is RowLinear else None,
                "lora_minimum_columns": MINIMUM_COLUMNS if self.lora is ColumnLoRALinear else None,
                "lora_column_padding": "repeat single column; discard repeated output" if self.lora is ColumnLoRALinear else None,
                "attention_query_tokens": QUERY_TOKENS if self.attention is QueryAttention else None,
                "attention_query_padding": "repeat final query/mask; discard repeated outputs" if self.attention is QueryAttention else None,
                "caches": [qualified(kind) for kind in expected],
                "batch_kv": qualified(mlx_cache.TypedBatchKVCache),
                "recurrence": {"module": qualified(mlx_recurrence.CheckpointedDeltaNet),
                               "segment_tokens": mlx_recurrence.CHECKPOINT_TOKENS},
                "packages": {name: version(name) for name in ("mlx", "mlx-lm")},
                **({"projection_stock_rows": SPLIT_ROWS, "prefill": "one request per prefill call"} if self.linear is RowLinear else {}),
                **({"attention_stock_queries": SPLIT_QUERIES} if self.attention is QueryAttention else {})}


PRIMARY = Profile(name="independent-native-rows/v8", linear=RowLinear, lora=ColumnLoRALinear,
                  attention=QueryAttention)
NATIVE = Profile(name="native-library-arithmetic/v4", linear=nn.QuantizedLinear, lora=LoRALinear, attention=Qwen3NextAttention)
