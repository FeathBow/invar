from dataclasses import replace
from typing import NamedTuple

import torch
from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend, GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

from worker.vllm.recurrence import recurrent

FORMAT = "invar-gdn-recurrent-f32-v1"


class Inputs(NamedTuple):
    mixed_qkv: torch.Tensor
    b: torch.Tensor
    a: torch.Tensor
    core_attn_out: torch.Tensor


class MetadataBuilder(GDNAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        if self.use_spec_decode:
            raise ValueError("The native Invar GDN recurrence does not implement speculative state branching")
        native = super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)
        metadata = compute_causal_conv1d_metadata(common_attn_metadata.query_start_loc_cpu,
                                                 device=common_attn_metadata.query_start_loc.device)
        return replace(native, has_initial_state=common_attn_metadata.compute_num_computed_tokens() > 0,
                       nums_dict=metadata[0], batch_ptr=metadata[1], token_chunk_offset_ptr=metadata[2])


class Backend(GDNAttentionBackend):
    @staticmethod
    def get_name():
        return "INVAR_GDN_RECURRENT"

    @staticmethod
    def supports_batch_invariance():
        return True

    @staticmethod
    def get_builder_cls():
        return MetadataBuilder


def core(layer, inputs):
    metadata = get_forward_context().attn_metadata
    if metadata is None:
        return
    metadata = metadata[layer.prefix]
    if metadata.spec_sequence_masks is not None:
        raise ValueError("The native Invar GDN recurrence cannot consume speculative metadata")
    state = layer.kv_cache[0] if is_conv_state_dim_first() else layer.kv_cache[0].transpose(-1, -2)
    convolved = causal_conv1d_fn(
        x=inputs.mixed_qkv[:metadata.num_actual_tokens].transpose(0, 1),
        weight=layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)),
        bias=layer.conv1d.bias, activation=layer.activation, conv_states=state,
        query_start_loc=metadata.non_spec_query_start_loc, cache_indices=metadata.non_spec_state_indices_tensor,
        has_initial_state=metadata.has_initial_state, metadata=metadata,
    ).transpose(0, 1)
    recurrent(layer, inputs, convolved=convolved, metadata=metadata)


class Layer(QwenGatedDeltaNetAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not current_platform.is_cuda() or not envs.VLLM_BATCH_INVARIANT:
            raise ValueError("The Invar GDN extension requires the selected CUDA batch-invariant engine profile")
        if self.num_spec or self.get_state_dtype()[1] != torch.float32:
            raise ValueError("The Invar GDN recurrence requires non-speculative FP32 recurrent state")
        # Keep the native projections and output norm; all core updates use core().
        self.enable_fused_gdn_decode = False
        init_logger(__name__).info_once("Using %s for both GDN prefill and decode", FORMAT)

    def _forward_core(self, *args, **kwargs):
        return core(self, Inputs(*args, **kwargs))


def install():
    QwenGatedDeltaNetAttention.register_oot(Layer)
    register_backend(MambaAttentionBackendEnum.GDN_ATTN, "vllm_gdn.Backend", is_mamba=True)
