import struct

import torch

from worker.advantage import word
from worker import record as probability_record
from worker.record import FORMAT, ROLES, Observation, Evaluated, loss, document, save

ENCODINGS = {torch.float32: ("F32", torch.int32, (1 << 32) - 1),
             torch.float64: ("F64", torch.int64, (1 << 64) - 1)}


def words(tensor):
    _, integer, mask = ENCODINGS[tensor.dtype]
    return tuple(value & mask for value in tensor.detach().contiguous().view(integer).cpu().tolist())


def capture(sample, tokens):
    dtype, _, _ = ENCODINGS[tokens.current.dtype]
    return Observation(sample=sample, dtype=dtype,
                       words=tuple(words(getattr(tokens, role)) for role in ROLES),
                       active=tuple(tokens.active.detach().cpu().tolist()))


def checked(sample, tokens, *, advantage, count):
    if count <= 0 or tokens.active.dtype != torch.bool or tokens.active.shape != (count,):
        raise ValueError("Objective mask must describe every admitted response token")
    for role in ROLES:
        tensor = getattr(tokens, role)
        if tensor.dtype != torch.float32 or tensor.shape != (count,) or tensor.device != tokens.active.device:
            raise ValueError("Admitted objective inputs must be matching FP32 response vectors")
    observed = capture(sample, tokens)
    if not all(observed.active):
        raise ValueError("Every admitted response token must be active")
    if observed.words[ROLES.index("advantage")] != (word(advantage),) * count:
        raise ValueError("Actual objective advantage differs from the checked sample advantage")
    return observed


def cotangents(observed, profile, *, total, device, linearized):
    return probability_record.cotangents(observed, profile, total=total, linearized=words(linearized),
                                         materialize=lambda values: tensor(values, device=device),
                                         check=lambda expected, actual: checked_cotangents(expected, actual, device=device))


def checked_cotangents(expected, tensors, *, device):
    for value, intended in zip(tensors, expected, strict=True):
        if value.dtype != torch.float32 or value.shape != (len(intended),) or value.device != device:
            raise ValueError("Actual scalar cotangents must be FP32 vectors on the model device")
    actual = tuple(words(value) for value in tensors)
    if actual != expected:
        raise ValueError("Actual scalar cotangents differ from the calculated FP32 words")
    return actual


def tensor(encoded, *, device):
    storage = bytearray(struct.pack(f"={len(encoded)}I", *encoded))
    return torch.frombuffer(storage, dtype=torch.float32).clone().to(device)


def logprobs(model, trajectory, *, device="cuda"):
    tokens = trajectory.tokens.to(device)
    response = tokens[:, trajectory.prompt_length:]
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False,
                   logits_to_keep=response.shape[1] + 1).logits[:, :-1, :].float()
    selected = torch.log_softmax(logits / trajectory.request.temperature, dim=-1).gather(-1, response.unsqueeze(-1)).reshape(-1)
    if not selected.isfinite().all():
        raise RuntimeError("Non-finite selected model log probability")
    return selected

