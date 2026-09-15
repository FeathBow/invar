from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch

from worker.vllm.identity import attributes, base, canonical, digest, kind
from worker.vllm.lora import TRANSFORM
from worker.vllm.quantization import observe as quantization

CONFIGURATION_FIELDS = {
    "model_config": ("dtype", "max_model_len", "quantization", "quantization_config", "enforce_eager",
                     "logprobs_mode", "use_fp64_gumbel", "disable_sliding_window", "disable_cascade_attn",
                     "model_impl", "language_model_only"),
    "cache_config": ("block_size", "cache_dtype", "sliding_window", "enable_prefix_caching",
                     "kv_cache_dtype_skip_layers", "mamba_block_size", "mamba_cache_dtype",
                     "mamba_ssm_cache_dtype", "mamba_cache_mode", "use_replayssm", "kv_sharing_fast_prefill"),
    "compilation_config": ("mode", "backend", "custom_ops", "ir_enable_torch_wrap", "splitting_ops",
                           "compile_sizes", "compile_ranges_endpoints", "inductor_compile_config",
                           "cudagraph_mode", "cudagraph_capture_sizes", "cudagraph_specialize_lora",
                           "use_inductor_graph_partition", "pass_config", "dynamic_shapes_config",
                           "enabled_custom_ops", "disabled_custom_ops"),
    "parallel_config": ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                        "prefill_context_parallel_size", "decode_context_parallel_size",
                        "disable_custom_all_reduce", "enable_dbo", "rank", "world_size"),
    "scheduler_config": ("max_num_batched_tokens", "max_num_scheduled_tokens", "max_num_seqs",
                         "enable_chunked_prefill", "policy", "disable_hybrid_kv_cache_manager",
                         "scheduler_reserve_full_isl", "prefill_schedule_interval", "async_scheduling"),
}
MODULE_FIELDS = ("eps", "epsilon", "variance_epsilon", "hidden_size", "intermediate_size", "head_size",
                 "num_heads", "num_kv_heads", "scale", "scaling", "softmax_scale", "logits_soft_cap",
                 "sliding_window", "is_causal", "rotary_dim", "max_position_embeddings", "base",
                 "mrope_section", "mrope_interleaved", "activation", "use_bias", "input_size", "output_size",
                 "output_sizes", "output_partition_sizes", "output_slices", "tp_rank", "tp_size")
METHOD_FIELDS = ("forward", "_forward_method", "forward_qkv", "forward_core", "apply")
SOURCE_MODULES = ("vllm_identity", "vllm_profile", "vllm_quantization", "vllm_lora", "vllm_rollout", "vllm_worker",
                  "vllm_mapping", "vllm_execution", "vllm_execution_state", "vllm_gdn", "gdn_kernel",
                  "vllm.v1.engine.core_client", "vllm.v1.engine.core", "vllm.v1.engine.llm_engine",
                  "vllm.v1.core.sched.scheduler", "vllm.v1.core.kv_cache_manager",
                  "vllm.lora.model_manager", "vllm.lora.worker_manager",
                  "vllm.config.lora", "vllm.lora.ops.triton_ops.kernel_utils",
                  "vllm.lora.ops.triton_ops.lora_shrink_op", "vllm.lora.ops.triton_ops.lora_expand_op")
ENTRY_FILES = (
    "vllm/configuration.py",
    "vllm/runtime.py",
    "vllm/entry.py",
    "vllm/infer.py",
    "vllm/session.py",
    "vllm/resident.py",
    "vllm/residency.py",
    "dispatch.py",
    "resident.py",
    "vllm/batch.py",
    "vllm/inspect.py",
    "batch.py",
    "report.py",
    "hf/infer.py",
    "hf/session.py",
    "hf/operation.py",
    "tokenization.py",
    "hf/decoding.py",
    "invocation.py",
    "registry.py",
    "hf/metrics.py",
    "cohort.py",
    "core.py",
    "hf/artifact.py",
    "hf/policy.py",
    "hf/tensors.py",
    "hf/rollout.py",
    "hf/frozen.py",
)


@dataclass(frozen=True, kw_only=True)
class Loaded:
    adapter_id: int
    base: str
    assembly: str


def callable_description(value, sources):
    function = getattr(value, "__func__", value)
    module = inspect.getmodule(function)
    if module is None or not getattr(module, "__file__", None):
        raise TypeError("Native assembly callable has no observable source module")
    sources[module.__name__] = module
    return {"module": function.__module__, "name": function.__qualname__}


def component(value, sources):
    cls = type(value)
    sources[cls.__module__] = sys.modules[cls.__module__]
    methods = {name: callable_description(getattr(value, name), sources)
               for name in METHOD_FIELDS if callable(getattr(value, name, None))}
    settings = {name: canonical(getattr(value, name)) for name in MODULE_FIELDS if hasattr(value, name)}
    result = {"class": kind(value), "methods": methods, "settings": settings}
    quantization = getattr(value, "quant_config", None)
    if quantization is not None:
        result["quantization"] = {"class": kind(quantization), "settings": canonical(vars(quantization))}
    return result


def modules(model, sources):
    result = {}
    for name, module in model.named_modules(remove_duplicate=False):
        described = component(module, sources)
        for field in ("quant_method", "impl"):
            value = getattr(module, field, None)
            if value is not None:
                described[field] = component(value, sources)
        result[name] = described
    return result


def configuration(runner):
    config = runner.vllm_config
    result = {name: attributes(getattr(config, name), fields) for name, fields in CONFIGURATION_FIELDS.items()}
    result.update({name: canonical(getattr(config, name)) for name in ("attention_config", "kernel_config", "lora_config")})
    result["hf_config"] = canonical(config.model_config.hf_config.to_dict())
    result["hf_text_config"] = canonical(config.model_config.hf_text_config.to_dict())
    return result


def numerical():
    import vllm
    from vllm import envs

    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "vllm": vllm.__version__,
            "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "batch_invariant": envs.VLLM_BATCH_INVARIANT, "lora_dual_stream": envs.VLLM_LORA_ENABLE_DUAL_STREAM,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}


def description(runner, native):
    sources = {name: sys.modules[name] for name in SOURCE_MODULES}
    result = {"format": "invar-native-assembly-v1", "configuration": configuration(runner),
              "numerical": numerical(), "modules": modules(native.model, sources),
              "quantization": quantization(native.model),
              "sampler": component(runner.sampler, sources),
              "lora_buffers": {name: {factor: [{"dtype": str(value.dtype), "shape": list(value.shape)}
                                               for value in getattr(module, factor)]
                                      for factor in ("lora_a_stacked", "lora_b_stacked")}
                               for name, module in native.modules.items()}}
    result["sources"] = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                         for name, module in sources.items()}
    root = Path(__file__).resolve().parents[1]
    result["entry_sources"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ENTRY_FILES}
    return result


def inspection(runner, native, *, policies):
    frozen = base(native.model)
    assembled = description(runner, native)
    result = []
    for adapter_id, (package, targets) in policies.items():
        policy = {"configuration": json.loads(package.configuration), "targets": [asdict(value) for value in targets],
                  "transform": TRANSFORM,
                  "source_model": {name: package.source[name] for name in ("base", "assembly", "tokenizer")}}
        result.append(Loaded(adapter_id=adapter_id, base=frozen, assembly=digest({"native": assembled, "policy": policy})))
    return tuple(result), assembled


def observe(runner, native, *, policies):
    models, _ = inspection(runner, native, policies=policies)
    return models
