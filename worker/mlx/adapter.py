import hashlib
import json
import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

from worker.mlx import tensors as mlx_tensors

TARGET = re.compile(r"language_model\.model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
RANK = 8
SCALE = 2.0


def state(model):
    return dict(tree_flatten(model.trainable_parameters()))


def install(model, values):
    current = state(model)
    if current.keys() != values.keys() or any(current[name].shape != value.shape or value.dtype != mx.float32
                                             for name, value in values.items()):
        raise ValueError("Native adapter differs from the resolved trainable parameters")
    model.update(tree_unflatten(list(values.items())))
    mx.eval(model.trainable_parameters())
    if not mlx_tensors.equal(values, state(model)):
        raise RuntimeError("Native model did not consume the requested adapter tensors")


def targets(model, layers):
    if len(model.layers) != layers:
        raise ValueError("Native model differs from the declared layer inventory")
    selected = {name: module for name, module in model.named_modules() if TARGET.fullmatch(name)}
    actual = {(int(TARGET.fullmatch(name)[1]), TARGET.fullmatch(name)[2]) for name in selected}
    if actual != {(index, name) for index in range(layers) for name in PROJECTIONS}:
        raise ValueError("Native model is missing a declared LoRA projection")
    return selected


def quantization(module):
    if not isinstance(module, nn.QuantizedLinear) or (module.bits, module.group_size, module.mode) != (4, 64, "affine"):
        raise ValueError("Native LoRA requires the declared affine4/group64 linear representation")


def create(model, *, layers):
    selected = targets(model, layers)
    for module in selected.values():
        quantization(module)
    model.freeze()
    linear_to_lora_layers(model, num_layers=layers,
                         config={"rank": RANK, "scale": SCALE, "dropout": 0.0,
                                 "keys": ["mlp." + name for name in PROJECTIONS]})
    mx.eval(model.trainable_parameters())
    expected = {name + suffix for name in selected for suffix in (".lora_a", ".lora_b")}
    parameters = state(model)
    if parameters.keys() != expected or any(value.dtype != mx.float32 for value in parameters.values()):
        raise ValueError("Native trainable parameters differ from the complete FP32 LoRA inventory")
    return parameters


def assembly(model):
    selected = targets(model, len(model.layers))
    observed = {}
    for name, module in selected.items():
        if not isinstance(module, LoRALinear) or type(module.dropout) is not nn.Dropout:
            raise ValueError("Native adapter has an undeclared executable module")
        quantization(module.linear)
        if module.scale != SCALE or module.dropout._p_1 != 1 or module.lora_a.shape[-1] != RANK or module.lora_b.shape[0] != RANK:
            raise ValueError("Native LoRA scale, rank or dropout differs from the declared profile")
        observed[name] = {"type": type(module).__module__ + "." + type(module).__qualname__,
                          "scale": module.scale, "dropout_keep": module.dropout._p_1,
                          "a": list(module.lora_a.shape), "b": list(module.lora_b.shape),
                          "bits": module.linear.bits, "group_size": module.linear.group_size, "mode": module.linear.mode}
    return observed


def images(model, config, *, numerics):
    trainable = state(model)
    base = hashlib.sha256(json.dumps(["invar-mlx-base/v1", config], sort_keys=True, separators=(",", ":")).encode())
    for name, value in sorted(tree_flatten(model.parameters())):
        if name not in trainable:
            base.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            base.update(mlx_tensors.view(value))
    structure = {"format": "invar-mlx-assembly/v2", "layers": len(model.layers), "numerics": numerics,
                 "model_type": type(model).__module__ + "." + type(model).__qualname__, "adapter": assembly(model)}
    encoded = json.dumps(structure, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"base": base.hexdigest(), "assembly": hashlib.sha256(encoded).hexdigest()}
