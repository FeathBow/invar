import hashlib
import json

import torch
from peft.tuners.lora.layer import LoraLayer
from transformers import PreTrainedConfig

from worker.hf import backend
from worker.hf import decoding
from worker import implementation

FORMAT = "invar-model-assembly-v4"
LOCATION_FIELDS = {"_name_or_path", "base_model_name_or_path"}
BACKENDS = ("_attn_implementation", "_experts_implementation")
LORA_SETTINGS = ("r", "lora_alpha", "scaling", "use_dora", "use_rslora", "lora_bias",
                 "ephemeral_gpu_offload", "cast_input_dtype_enabled")
MODULE_SETTINGS = ("compute_dtype", "quant_type", "quant_storage", "fan_in_fan_out", "is_target_conv_1d_layer",
                   "gradient_checkpointing")


def kind(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def encode(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"Unsupported model assembly value: {kind(value)}")


def configuration(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return portable(value)
    declared = portable(value.to_dict())
    nested = {name: configuration(item) for name, item in vars(value).items()
              if isinstance(item, PreTrainedConfig)}
    actual = {name: getattr(value, name) for name in BACKENDS if hasattr(value, name)}
    return {"class": kind(value), "declared": declared, "nested": nested, "backends": actual}


def portable(value):
    if isinstance(value, dict):
        return {name: portable(item) for name, item in value.items() if name not in LOCATION_FIELDS}
    if isinstance(value, (tuple, list)):
        return [portable(item) for item in value]
    return value


def dropout(value):
    if isinstance(value, torch.nn.Identity):
        return {"class": kind(value)}
    if isinstance(value, torch.nn.Dropout):
        return {"class": kind(value), "p": value.p, "inplace": value.inplace}
    raise TypeError(f"Unsupported LoRA dropout module: {kind(value)}")


def layer(value):
    return {"settings": {name: getattr(value, name) for name in LORA_SETTINGS},
            "active": list(value.active_adapters), "disabled": value.disable_adapters,
            "merged": list(value.merged_adapters),
            "dropout": {name: dropout(item) for name, item in value.lora_dropout.items()},
            "variants": {name: kind(item) for name, item in value.lora_variant.items()}}


def parameters(model):
    return {name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in model.named_parameters(remove_duplicate=False) if value.requires_grad}


def description(model, role):
    modules = dict(model.named_modules(remove_duplicate=False))
    generation = {"generation": decoding.description()} if role == implementation.INFERENCE else {}
    return {"format": FORMAT, "role": role, "numerical": backend.description(), "implementation": implementation.current(role),
            **generation,
            "classes": {name: kind(value) for name, value in modules.items()},
            "parameters": parameters(model),
            "module_settings": {name: {key: getattr(value, key) for key in MODULE_SETTINGS if hasattr(value, key)}
                                for name, value in modules.items()},
            "model": configuration(getattr(model.get_base_model(), "config", None)),
            "adapters": {name: configuration(value) for name, value in model.peft_config.items()},
            "layers": {name: layer(value) for name, value in modules.items() if isinstance(value, LoraLayer)}}


def digest(model, role):
    encoded = json.dumps(description(model, role), default=encode, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify(model, expected, role):
    if not isinstance(expected, str) or expected != digest(model, role):
        raise RuntimeError("Checkpoint model assembly binding mismatch")
