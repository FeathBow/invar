import struct

import torch

ENCODINGS = {torch.float32: ("F32", torch.int32, (1 << 32) - 1),
             torch.float64: ("F64", torch.int64, (1 << 64) - 1)}


def words(tensor):
    _, integer, mask = ENCODINGS[tensor.dtype]
    return tuple(value & mask for value in tensor.detach().contiguous().view(integer).cpu().tolist())


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

