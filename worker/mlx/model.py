from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import wired_limit
from mlx_lm.utils import load_model

from worker.cohort import fields
from worker import core
from worker.mlx import adapter as mlx_adapter
from worker.mlx import numerics as mlx_numerics
from worker.mlx.rollout import Sampling
from worker.mlx import recurrence as mlx_recurrence
from worker.mlx import tensors as mlx_tensors
from worker.mlx import tokenization as mlx_tokenization
from worker.hf import operation
from worker.implementation import INFERENCE, LEARNING
from worker.mlx import implementation as mlx_implementation
from worker.mlx import training as mlx_training

MODEL = "mlx-community/Qwen3.8-27B-4bit"
REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
LAYERS = 64
DEFAULT_SEED = 17
CACHE_BYTES = 256 * 1024 * 1024
PREFILL_STEP = 512


@dataclass(frozen=True, kw_only=True)
class Configuration:
    batch_size: int = 1
    prefill_step: int = PREFILL_STEP
    cache_bytes: int = CACHE_BYTES

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.batch_size, self.prefill_step)) or type(self.cache_bytes) is not int or self.cache_bytes < 0:
            raise ValueError("Native batch/prefill sizes must be positive integers and the free-buffer cache nonnegative")

    def sampling(self):
        return Sampling(batch_size=self.batch_size, prefill_step=self.prefill_step)


@dataclass(frozen=True, kw_only=True)
class Loaded:
    model: object
    tokenizer: object
    config: dict
    identity: tuple[str, str]
    numerics: mlx_numerics.Profile = mlx_numerics.PRIMARY


def configuration(path):
    if path is None:
        return Configuration()
    value = core.decode(Path(path).read_text())
    fields(value, "format batch_size prefill_step cache_bytes")
    if value["format"] != "invar-mlx-runtime-v1":
        raise ValueError("Expected native MLX runtime configuration")
    return Configuration(**{name: item for name, item in value.items() if name != "format"})


def resolve(cache):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache, local_files_only=True))


def load(cache, *, scope, configuration, measure, emit, seed=DEFAULT_SEED, initial=None, numerics=mlx_numerics.PRIMARY, identified=False):
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("The native MLX Metal device is required")
    previous = mx.set_cache_limit(configuration.cache_bytes)
    scope.callback(mx.set_cache_limit, previous)
    path = resolve(cache)
    mx.random.seed(seed)

    def prepare():
        emit("loading", {"model": MODEL, "revision": REVISION})
        model, config = load_model(path, lazy=False, strict=True)
        if len(model.layers) != LAYERS or config["text_config"]["num_hidden_layers"] != LAYERS:
            raise ValueError("The native loader requires the complete declared model")
        mlx_adapter.create(model, layers=LAYERS)
        numerics.install(model)
        model.eval()
        scope.enter_context(wired_limit(model))
        loaded = Loaded(model=model, tokenizer=operation.load(path), config=config, identity=(MODEL, REVISION), numerics=numerics)
        if initial is not None:
            activate(loaded, initial[0], expected=initial[1])
        reported = profile(configuration, numerics=numerics.name)
        if identified:
            reported["inference"] = mlx_adapter.images(model, config, numerics=described(loaded, INFERENCE))
        emit("profile", reported)
        return loaded

    return measure("load", prepare)


load_native = partial(load, numerics=mlx_numerics.NATIVE)


def profile(configuration, *, numerics):
    return {"model": MODEL, "revision": REVISION,
            "native": {"device": mx.device_info()["device_name"],
                       "numerics": numerics,
                       "packages": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
                       "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
                       "adapter": {"layers": LAYERS, "targets": list(mlx_adapter.PROJECTIONS),
                                   "rank": mlx_adapter.RANK, "scale": mlx_adapter.SCALE, "dtype": "mlx.core.float32"},
                       "batch_size": configuration.batch_size, "prefill_step": configuration.prefill_step,
                       "free_buffer_cache_bytes": configuration.cache_bytes,
                       "wired_limit_bytes": mx.device_info()["max_recommended_working_set_size"],
                       "gradient_checkpointing": {"decoder": "materialized inputs and layerwise native VJPs",
                                                  "primal": "native VJP trace before checked scalar cotangents",
                                                  "recurrence": "native gated-delta ops with differentiable segment state",
                                                  "recurrent_vjp": "explicit prefix/segments/suffix; shared normalization differentiated once",
                                                  "segment_tokens": mlx_recurrence.CHECKPOINT_TOKENS},
                       "optimizer_bias_correction": True}}


def described(loaded, role):
    observed = loaded.numerics.observe(loaded.model)
    learning = mlx_training.description(loaded.numerics) if role == LEARNING else {}
    return {**observed, **learning, "role": role, "implementation": mlx_implementation.current(role)}


def identities(loaded, role):
    numerical = described(loaded, role)
    return {"adapter": mlx_tensors.digest(mlx_adapter.state(loaded.model)),
            "tokenizer": mlx_tokenization.digest(loaded.tokenizer),
            **mlx_adapter.images(loaded.model, loaded.config, numerics=numerical)}


def assembly(loaded, role):
    return mlx_adapter.assembled(loaded.model, numerics=described(loaded, role))


def verify(loaded, expected, role):
    actual = identities(loaded, role)
    if actual != expected:
        raise ValueError("Actual native model, policy or tokenizer differs from its requested materialization")
    return actual


def activate(loaded, path, *, expected):
    mlx_adapter.install(loaded.model, mlx_tensors.policy(path, expected["adapter"]))
    verify(loaded, expected, INFERENCE)
    return loaded
