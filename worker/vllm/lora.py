import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch

from worker.hf.tensors import assert_equal, digest

FORMAT = "invar-vllm-lora-materialization-v1"
TRANSFORM = "fp32-b-times-scaling/packed-zero-padded/v1"
PEFT_PREFIX = "base_model.model."
FACTORS = ("A", "B")


@dataclass(frozen=True, kw_only=True)
class Target:
    source: str
    loaded: str
    resident: str
    index: int


@dataclass(frozen=True, kw_only=True)
class Consumption:
    package: str
    policy: str
    adapter_id: int
    slot: int
    targets: str
    transformed: str
    resident: str
    format: str = FORMAT
    transform: str = TRANSFORM


def qwen_mlp_targets(*, layers, source_prefix, runtime_prefix):
    if type(layers) is not int or layers <= 0:
        raise ValueError("Qwen layer count must be a positive integer")
    return tuple(Target(source=f"{PEFT_PREFIX}{source_prefix}.{layer}.mlp.{projection}_proj",
                        loaded=f"{runtime_prefix}.{layer}.mlp.{projection}_proj",
                        resident=f"{runtime_prefix}.{layer}.mlp.{packed}_proj", index=index)
                 for layer in range(layers)
                 for projection, packed, index in (("gate", "gate_up", 0),
                                                   ("up", "gate_up", 1), ("down", "down", 0)))


def factor_key(target, factor):
    return f"{target.source}.lora_{factor}.weight"


def patterned(configuration, field, name, *, default):
    for pattern, value in configuration.get(field, {}).items():
        if re.match(rf"(.*\.)?({pattern})$", name):
            return value
    return default


def layer_settings(configuration, target):
    name = target.source.removeprefix(PEFT_PREFIX)
    rank = patterned(configuration, "rank_pattern", name, default=configuration["r"])
    alpha = patterned(configuration, "alpha_pattern", name, default=configuration["lora_alpha"])
    if type(rank) is not int or rank <= 0:
        raise ValueError(f"Invalid LoRA rank: {name}")
    if type(alpha) not in (int, float) or not math.isfinite(alpha):
        raise ValueError(f"Invalid LoRA alpha: {name}")
    scaling = alpha / (math.sqrt(rank) if configuration.get("use_rslora", False) else rank)
    if not math.isfinite(scaling):
        raise ValueError(f"Nonfinite LoRA scaling: {name}")
    return rank, alpha, scaling


def grouped(targets):
    groups = defaultdict(list)
    for target in targets:
        groups[target.resident].append(target)
    return {name: tuple(sorted(values, key=lambda target: target.index))
            for name, values in groups.items()}


def target_groups(package, targets):
    package.verify()
    if not targets or len({target.source for target in targets}) != len(targets):
        raise ValueError("LoRA source targets must be nonempty and unique")
    if len({target.loaded for target in targets}) != len(targets):
        raise ValueError("LoRA loader targets must be unique")
    expected_keys = {factor_key(target, factor) for target in targets for factor in FACTORS}
    if package.tensors.keys() != expected_keys:
        raise ValueError("PEFT tensors do not cover the complete LoRA target map")
    return grouped(targets)


def native_layout(manager, name, targets):
    module = manager.modules[name]
    if manager.model.get_submodule(name) is not module:
        raise ValueError(f"LoRA manager does not own the executed module: {name}")
    if module.tp_size != 1 or module.tp_rank != 0 or manager.lora_config.fully_sharded_loras:
        raise ValueError("This LoRA materialization profile requires unsharded TP=1")
    if tuple(target.index for target in targets) != tuple(range(len(targets))):
        raise ValueError(f"LoRA slice map is incomplete or duplicated: {name}")
    if manager.packed_modules.get(name, [name]) != [target.loaded for target in targets]:
        raise ValueError(f"Native packed LoRA ordering mismatch: {name}")
    sizes = (len(module.lora_a_stacked), len(module.lora_b_stacked), module.n_slices)
    if sizes != (len(targets),) * len(sizes):
        raise ValueError(f"Native LoRA buffer count mismatch: {name}")


def layout(manager, package, targets):
    groups = target_groups(package, targets)
    if manager.modules.keys() != groups.keys():
        raise ValueError("Native LoRA modules differ from the complete target map")
    if manager.lora_config.lora_dtype != torch.float32:
        raise ValueError("The materialized LoRA profile requires actual FP32 weights")
    for name, values in groups.items():
        native_layout(manager, name, values)
    return groups


def selected(declared, name):
    if isinstance(declared, str):
        return re.fullmatch(declared, name) is not None
    return any(name == item or name.endswith("." + item) for item in declared)


def configured_targets(configuration, targets):
    declared = configuration["target_modules"]
    names = tuple(target.source.removeprefix(PEFT_PREFIX) for target in targets)
    if not all(selected(declared, name) for name in names):
        raise ValueError("LoRA target is absent from PEFT configuration")
    if not isinstance(declared, str):
        if any(not any(selected([item], name) for name in names) for item in declared):
            raise ValueError("PEFT configuration declares targets absent from the materialization map")


def configuration(manager, package, targets):
    from vllm.lora.peft_helper import PEFTHelper

    value = json.loads(package.configuration)
    if value.get("peft_type") != "LORA":
        raise ValueError("Native materialization requires a LoRA configuration")
    for feature in ("fan_in_fan_out", "lora_bias", "use_qalora", "alora_invocation_tokens", "target_parameters"):
        if value.get(feature):
            raise ValueError(f"The native linear LoRA materializer does not implement {feature}")
    configured_targets(value, targets)
    helper = PEFTHelper.from_dict(value)
    helper.validate_legal(manager.lora_config)
    return value, helper


def expected_factors(package, target, settings):
    rank, _, scaling = settings
    a, b = (package.tensors[factor_key(target, factor)] for factor in FACTORS)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
        raise ValueError(f"PEFT tensor dimensions differ from LoRA rank: {target.source}")
    scaled = b * scaling
    if not scaled.isfinite().all():
        raise ValueError(f"Scaled LoRA B is nonfinite: {target.source}")
    return a, scaled


def create(manager, package, *, targets, adapter_id):
    from vllm.lora.lora_model import LoRAModel

    configured, helper = configuration(manager, package, targets)
    settings = {target.loaded: layer_settings(configured, target) for target in targets}
    for target in targets:
        expected_factors(package, target, settings[target.loaded])
    rank = max(item[0] for item in settings.values())
    if rank > manager.lora_config.max_lora_rank:
        raise ValueError("Resolved PEFT rank exceeds the native LoRA buffer rank")
    # Native packing scales B in place. It must never alias the source package.
    model = LoRAModel.from_lora_tensors(
        adapter_id, {name: value.clone() for name, value in package.tensors.items()}, helper,
        device="cpu", dtype=torch.float32,
        weights_mapper=getattr(manager.model, "hf_to_vllm_mapper", None))
    if model.loras.keys() != settings.keys():
        raise ValueError("Native loader names differ from the explicit target map")
    model.rank = rank
    for target in targets:
        layer = model.loras[target.loaded]
        for factor, actual in zip(FACTORS, (layer.lora_a, layer.lora_b)):
            assert_equal(package.tensors[factor_key(target, factor)], actual)
        layer.rank, layer.lora_alpha, layer.scaling = settings[target.loaded]
    return model


def native_slices(layer):
    if layer.is_packed:
        return layer.lora_a, layer.lora_b, layer.scaling, layer.lora_alphas
    return [layer.lora_a], [layer.lora_b], [layer.scaling], [layer.lora_alpha]


def cached_slice(package, target, *, native, settings):
    a, b, scales, alphas = native
    if scales[target.index] != 1 or alphas[target.index] != settings[1]:
        raise ValueError(f"Native cached LoRA scaling mismatch: {target.loaded}")
    result = {}
    for factor, values, expected in zip(FACTORS, (a, b), expected_factors(package, target, settings)):
        assert_equal(expected, values[target.index])
        # Keep the independent source-derived values across native activation.
        result[f"{target.resident}.{target.index}.{factor}"] = expected
    return result


def cached_layer(layer, package, *, targets, configured):
    if layer.is_packed != (len(targets) > 1):
        raise ValueError(f"Native LoRA packing mismatch: {targets[0].resident}")
    native = native_slices(layer)
    if any(len(values) != len(targets) for values in native):
        raise ValueError(f"Native cached LoRA slice count mismatch: {targets[0].resident}")
    settings = tuple(layer_settings(configured, target) for target in targets)
    if layer.rank != settings[0][0]:
        raise ValueError(f"Native cached LoRA rank mismatch: {targets[0].resident}")
    result = {}
    for target, configured_layer in zip(targets, settings):
        result.update(cached_slice(package, target, native=native, settings=configured_layer))
    return result, tuple(value[0] for value in settings)


def cached(manager, package, *, groups, adapter_id):
    model = manager.get_adapter(adapter_id)
    if model is None or model.id != adapter_id or model.loras.keys() != groups.keys():
        raise ValueError("Native LoRA cache differs from the requested adapter or targets")
    configured, _ = configuration(manager, package, tuple(target for values in groups.values() for target in values))
    result = {}
    ranks = []
    for name, targets in groups.items():
        state, layer_ranks = cached_layer(model.loras[name], package, targets=targets, configured=configured)
        result.update(state)
        ranks.extend(layer_ranks)
    if model.rank != max(ranks):
        raise ValueError("Native cached adapter rank mismatch")
    return result


def resident_tensor(buffer, value, *, slots, rank, slot, factor, key):
    if buffer.dtype != torch.float32 or buffer.ndim != 4 or buffer.shape[1] != 1:
        raise ValueError(f"Native LoRA buffer representation mismatch: {key}")
    if buffer.shape[0] != slots:
        raise ValueError(f"Native LoRA slot count mismatch: {key}")
    actual = buffer[slot, 0].detach().cpu()
    value = value.detach().cpu()
    shape = (rank, value.shape[1]) if factor == "A" else (value.shape[0], rank)
    if tuple(actual.shape) != shape or any(wanted > available for wanted, available in zip(value.shape, shape)):
        raise ValueError(f"Native LoRA buffer dimensions mismatch: {key}")
    expected = torch.zeros_like(actual)
    expected[:value.shape[0], :value.shape[1]] = value
    assert_equal(expected, actual)
    return actual


def resident(manager, *, groups, transformed, slot):
    result = {}
    for name, targets in groups.items():
        module = manager.modules[name]
        for target in targets:
            for factor, buffers in zip(FACTORS, (module.lora_a_stacked, module.lora_b_stacked)):
                key = f"{name}.{target.index}.{factor}"
                result[key] = resident_tensor(buffers[target.index], transformed[key], slots=manager.lora_slots,
                                               rank=manager.lora_config.max_lora_rank, slot=slot, factor=factor, key=key)
        if tuple(module.output_slices) != tuple(buffer.shape[2] for buffer in module.lora_b_stacked):
            raise ValueError(f"Native LoRA output offsets mismatch: {name}")
    return result


def observation(manager, package, *, groups, targets, adapter_id, transformed):
    slots = [index for index, value in enumerate(manager.lora_index_to_id) if value == adapter_id]
    if len(slots) != 1 or adapter_id not in manager._active_adapters:
        raise ValueError("Requested LoRA is not bound to one active native slot")
    observed = resident(manager, groups=groups, transformed=transformed, slot=slots[0])
    mapping = json.dumps([asdict(target) for target in targets], sort_keys=True, separators=(",", ":"))
    return Consumption(package=package.identity, policy=package.source["adapter"], adapter_id=adapter_id,
                       slot=slots[0], targets=hashlib.sha256(mapping.encode()).hexdigest(),
                       transformed=digest(transformed), resident=digest(observed))


def verify(manager, package, *, targets, adapter_id):
    groups = layout(manager, package, targets)
    transformed = cached(manager, package, groups=groups, adapter_id=adapter_id)
    return observation(manager, package, groups=groups, targets=targets, adapter_id=adapter_id, transformed=transformed)


def activate(manager, package, *, targets, adapter_id):
    if type(adapter_id) is not int or adapter_id <= 0:
        raise ValueError("Native LoRA adapter ID must be a positive integer")
    groups = layout(manager, package, targets)
    if manager.get_adapter(adapter_id) is None:
        model = create(manager, package, targets=targets, adapter_id=adapter_id)
        if not manager.add_adapter(model):
            raise RuntimeError("Native LoRA registration did not install the requested adapter")
    transformed = cached(manager, package, groups=groups, adapter_id=adapter_id)
    manager.activate_adapter(adapter_id)
    return observation(manager, package, groups=groups, targets=targets, adapter_id=adapter_id, transformed=transformed)
