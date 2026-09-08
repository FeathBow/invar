import re

import torch
from safetensors.torch import load_file

from tensors import assert_equal, digest


def read_adapter(path, expected):
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Expected adapter identity must be a lowercase SHA-256 digest")
    state = load_file(path, device="cpu")
    if not state or any(value.dtype != torch.float32 or not value.isfinite().all()
                        for value in state.values()):
        raise ValueError("Adapter must contain finite FP32 tensors")
    if digest(state) != expected:
        raise ValueError("Adapter contents do not match the requested tensor identity")
    return state


def validate_schema(expected, received):
    if expected.keys() != received.keys():
        raise ValueError("Adapter keys do not match the resolved model targets")
    for name, value in expected.items():
        actual = received[name]
        if value.shape != actual.shape or value.dtype != actual.dtype:
            raise ValueError(f"Adapter tensor metadata mismatch: {name}")


def activate(model, state, *, base, assembly):
    from peft import set_peft_model_state_dict
    from probe import adapter_state
    import assembly as model_assembly
    import frozen

    model_assembly.verify(model, assembly)
    frozen.verify(model, base)
    validate_schema(adapter_state(model), state)
    set_peft_model_state_dict(model, state)
    loaded = adapter_state(model)
    assert_equal(state, loaded)
    return digest(loaded)
