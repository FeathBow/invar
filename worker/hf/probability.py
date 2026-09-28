import hashlib
import json
import struct

import torch

FORMAT = "invar-probabilities-v4"
ENCODINGS = {torch.float32: ("F32", torch.int32, (1 << 32) - 1),
             torch.float64: ("F64", torch.int64, (1 << 64) - 1)}


def words(tensor):
    _, integer, mask = ENCODINGS[tensor.dtype]
    return tuple(value & mask for value in tensor.detach().contiguous().view(integer).cpu().tolist())


def document(order, update, *, invocation, request):
    samples = [{"sample": name, "dtype": "F32", "proximal": list(update.proximal[name]),
                "steps": [{"step": step, "current": list(current)} for step, sample, current in update.currents if sample == name]}
               for name in order]
    return {"format": FORMAT, "invocation": invocation, "request": request, "samples": samples}


def save(path, order, update, *, invocation, request):
    encoded = json.dumps(document(order, update, invocation=invocation, request=request), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    with path.open("xb") as output:
        output.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


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

