# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Recurrent update arithmetic adapted from vLLM's MIT-origin FLA
# fused_sigmoid_gating.py, Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
from vllm.triton_utils import tl, triton

VALUE_TILE = 32
WARPS = 4
STAGES = 3
QK_EPSILON = 1e-6
SOFTPLUS_THRESHOLD = 20.0


@triton.jit
def update(tensors, layout: tl.constexpr, arithmetic: tl.constexpr):
    QKV, A, B, A_LOG, DT, STATE, STARTS, SLOTS, CONTINUATION, OUTPUT = tensors
    QKV_STRIDE: tl.constexpr = layout[0]
    STATE_STRIDE: tl.constexpr = layout[1]
    HEADS: tl.constexpr = layout[2]
    VALUE_HEADS: tl.constexpr = layout[3]
    KEY: tl.constexpr = layout[4]
    VALUE: tl.constexpr = layout[5]
    BLOCK_KEY: tl.constexpr = layout[6]
    BLOCK_VALUE: tl.constexpr = layout[7]
    SCALE: tl.constexpr = arithmetic[0]
    EPSILON: tl.constexpr = arithmetic[1]
    SOFTPLUS_LIMIT: tl.constexpr = arithmetic[2]
    sequence_head = tl.program_id(1)
    sequence = sequence_head // VALUE_HEADS
    head = sequence_head % VALUE_HEADS
    key_head = head // (VALUE_HEADS // HEADS)
    begin = tl.load(STARTS + sequence).to(tl.int64)
    end = tl.load(STARTS + sequence + 1).to(tl.int64)
    slot = tl.load(SLOTS + sequence).to(tl.int64)
    if begin == end or slot <= 0:
        return
    keys = tl.arange(0, BLOCK_KEY)
    values = tl.program_id(0) * BLOCK_VALUE + tl.arange(0, BLOCK_VALUE)
    mask = (values[:, None] < VALUE) & (keys[None, :] < KEY)
    state_pointer = STATE + slot * STATE_STRIDE + head * VALUE * KEY + values[:, None] * KEY + keys[None, :]
    state = tl.zeros((BLOCK_VALUE, BLOCK_KEY), dtype=tl.float32)
    if tl.load(CONTINUATION + sequence):
        state = tl.load(state_pointer, mask=mask, other=0).to(tl.float32)
    decay_weight = tl.exp(tl.load(A_LOG + head).to(tl.float32))
    time_bias = tl.load(DT + head).to(tl.float32)
    for token in range(begin, end):
        base = QKV + token * QKV_STRIDE
        query = tl.load(base + key_head * KEY + keys, keys < KEY, 0).to(tl.float32)
        key = tl.load(base + HEADS * KEY + key_head * KEY + keys, keys < KEY, 0).to(tl.float32)
        value = tl.load(base + 2 * HEADS * KEY + head * VALUE + values, values < VALUE, 0).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query) + EPSILON)
        key *= tl.rsqrt(tl.sum(key * key) + EPSILON)
        query *= SCALE
        time = tl.load(A + token * VALUE_HEADS + head).to(tl.float32) + time_bias
        softplus = tl.where(time <= SOFTPLUS_LIMIT, tl.log(1 + tl.exp(time)), time)
        gate = tl.sigmoid(tl.load(B + token * VALUE_HEADS + head).to(tl.float32))
        state *= tl.exp(-decay_weight * softplus)
        value -= tl.sum(state * key[None, :], 1)
        value *= gate
        state += value[:, None] * key[None, :]
        result = tl.sum(state * query[None, :], 1)
        output_pointer = OUTPUT + (token * VALUE_HEADS + head) * VALUE + values
        tl.store(output_pointer, result.to(OUTPUT.dtype.element_ty), values < VALUE)
    # The FP32 cache is the same state boundary for any scheduler partition.
    tl.store(state_pointer, state, mask)


def recurrent(layer, inputs, *, convolved, metadata):
    heads = layer.num_k_heads // layer.tp_size
    value_heads = layer.num_v_heads // layer.tp_size
    key, value = layer.head_k_dim, layer.head_v_dim
    block_value = min(triton.next_power_of_2(value), VALUE_TILE)
    sequences = metadata.non_spec_query_start_loc.numel() - 1
    update[(triton.cdiv(value, block_value), sequences * value_heads)](
        tensors=(convolved, inputs.a.contiguous(), inputs.b.contiguous(), layer.A_log, layer.dt_bias,
                 layer.kv_cache[1], metadata.non_spec_query_start_loc, metadata.non_spec_state_indices_tensor,
                 metadata.has_initial_state, inputs.core_attn_out),
        layout=(convolved.stride(0), layer.kv_cache[1].stride(0), heads, value_heads, key, value,
                triton.next_power_of_2(key), block_value),
        arithmetic=(key ** -0.5, QK_EPSILON, SOFTPLUS_THRESHOLD),
        num_warps=WARPS, num_stages=STAGES,
    )
