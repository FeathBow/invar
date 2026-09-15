import re

import torch
from safetensors.torch import load_file

from worker.hf.tensors import assert_equal, digest


def read_adapter(path, expected):
    validate_identity(expected)
    state = load_file(path, device="cpu")
    validate_state(state, expected)
    return state


def validate_identity(expected):
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Expected adapter identity must be a lowercase SHA-256 digest")


def validate_state(state, expected):
    validate_identity(expected)
    if not state or any(value.dtype != torch.float32 or not value.isfinite().all()
                        for value in state.values()):
        raise ValueError("Adapter must contain finite FP32 tensors")
    if digest(state) != expected:
        raise ValueError("Adapter contents do not match the requested tensor identity")


def verify(model, expected, *, base, assembly):
    from worker.hf.probe import adapter_state
    from worker.hf import assembly as model_assembly
    from worker.hf import frozen

    model_assembly.verify(model, assembly)
    frozen.verify(model, base)
    validate_state(adapter_state(model), expected)
    return expected


def validate_schema(expected, received):
    if expected.keys() != received.keys():
        raise ValueError("Adapter keys do not match the resolved model targets")
    for name, value in expected.items():
        actual = received[name]
        if value.shape != actual.shape or value.dtype != actual.dtype:
            raise ValueError(f"Adapter tensor metadata mismatch: {name}")


def activate(model, state, *, base, assembly):
    from peft import set_peft_model_state_dict
    from worker.hf.probe import adapter_state
    from worker.hf import assembly as model_assembly
    from worker.hf import frozen

    model_assembly.verify(model, assembly)
    frozen.verify(model, base)
    validate_schema(adapter_state(model), state)
    set_peft_model_state_dict(model, state)
    loaded = adapter_state(model)
    assert_equal(state, loaded)
    return digest(loaded)
