from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version

import mlx.nn as nn
from mlx_lm.models.qwen3_next import Qwen3NextAttention
from mlx_lm.tuner.lora import LoRALinear

from worker.mlx import cache as mlx_cache
from worker.mlx.attention import QUERY_TOKENS, QueryAttention
from worker.mlx.projection import BITS, GROUP_SIZE, MINIMUM_COLUMNS, MODE, ColumnLoRALinear, RowLinear
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
    learning_linear: type | None = None

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

    @contextmanager
    def learning(self, model):
        if self.learning_linear is None:
            yield
            return
        modules = tuple(inventory(model, nn.QuantizedLinear).values())
        if not modules or any(type(module) is not self.linear for module in modules):
            raise TypeError("Native learning requires its declared inference projection inventory")
        try:
            for module in modules:
                object.__setattr__(module, "__class__", self.learning_linear)
            yield
            if any(type(module) is not self.learning_linear for module in modules):
                raise TypeError("Actual native learning projections changed inside their owned operation")
        finally:
            for module in modules:
                object.__setattr__(module, "__class__", self.linear)

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
        learning = {} if self.learning_linear is None else {
            "learning_projection": {"module": qualified(self.learning_linear),
                                    "schedule": "one complete logical trajectory per native projection",
                                    "roles": ["proximal", "reference", "current", "objective_vjp", "reward_vjp"],
                                    "lifetime": "owned numerical operation; inference classes restored before state observation"}}
        return {"format": "invar-mlx-numerics/v1", "name": self.name, "modules": observed, **learning,
                "lora_minimum_columns": MINIMUM_COLUMNS if self.lora is ColumnLoRALinear else None,
                "lora_column_padding": "repeat single column; discard repeated output" if self.lora is ColumnLoRALinear else None,
                "attention_query_tokens": QUERY_TOKENS if self.attention is QueryAttention else None,
                "attention_query_padding": "repeat final query/mask; discard repeated outputs" if self.attention is QueryAttention else None,
                "caches": [qualified(kind) for kind in expected],
                "batch_kv": qualified(mlx_cache.TypedBatchKVCache),
                "recurrence": {"module": qualified(mlx_recurrence.CheckpointedDeltaNet),
                               "segment_tokens": mlx_recurrence.CHECKPOINT_TOKENS},
                "packages": {name: version(name) for name in ("mlx", "mlx-lm")}}


PRIMARY = Profile(name="independent-native-rows/v3", linear=RowLinear, lora=ColumnLoRALinear,
                  attention=QueryAttention, learning_linear=nn.QuantizedLinear)
NATIVE = Profile(name="native-library-arithmetic/v1", linear=nn.QuantizedLinear, lora=LoRALinear, attention=Qwen3NextAttention)
