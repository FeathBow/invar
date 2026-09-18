from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json

import torch

from worker.hf.frozen import blocks

LOCATION_FIELDS = frozenset(("_name_or_path", "base_model_name_or_path"))
QUANT_STATE_FIELDS = ("absmax", "shape", "code", "dtype", "blocksize", "quant_type", "offset", "state2", "nested")
PARAMETER_STATE_FIELDS = ("bnb_quant_state", "bnb_shard_offsets", "quant_state", "pack_factor")
IDENTITY_WORKERS = 16


def kind(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def tensor(value):
    result = hashlib.sha256(b"invar-native-tensor-v1\0")
    for block in blocks(value):
        result.update(block)
    return {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": result.hexdigest()}


def mapping(value):
    if any(type(key) not in (str, int) for key in value):
        raise TypeError("Native identity requires string or integer mapping keys")
    selected = {key: item for key, item in value.items() if key not in LOCATION_FIELDS}
    result = {str(key): canonical(item) for key, item in selected.items()}
    if len(result) != len(selected):
        raise ValueError("Native identity mapping keys alias after encoding")
    return result


def structured(value):
    if isinstance(value, Mapping):
        return mapping(value)
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(item) for item in value), key=encoded)
    return object_state(value)


def object_state(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: canonical(getattr(value, field.name)) for field in fields(value)}
    if kind(value) == "bitsandbytes.functional.QuantState":
        return {name: canonical(getattr(value, name)) for name in QUANT_STATE_FIELDS}
    raise TypeError(f"Unsupported native identity value: {kind(value)}")


def canonical(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, torch.Tensor):
        return tensor(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Enum):
        return {"class": kind(value), "value": canonical(value.value)}
    return structured(value)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(canonical(value)).encode()).hexdigest()


def attributes(value, names):
    return {name: canonical(getattr(value, name)) for name in names}


def observed_fields(value, executor):
    return dict(zip(value, executor.map(canonical, value.values()), strict=True))


def base(model):
    # Native LoRA slots and request caches are not registered model state. Include
    # nonpersistent buffers and quantization auxiliaries that state_dict omits.
    parameters = dict(model.named_parameters(remove_duplicate=False))
    auxiliary = {name: {key: getattr(value, key) for key in PARAMETER_STATE_FIELDS if hasattr(value, key)}
                 for name, value in parameters.items()}
    # Hash independent tensor streams concurrently; every observation still reads
    # all current bytes and preserves each tensor's original SHA-256 sequence.
    with ThreadPoolExecutor(max_workers=IDENTITY_WORKERS) as executor:
        observed = {"auxiliary": observed_fields(auxiliary, executor),
                    "state": observed_fields(model.state_dict(), executor),
                    "buffers": observed_fields(dict(model.named_buffers(remove_duplicate=False)), executor)}
    return digest({"format": "invar-native-base-v1", **observed})
