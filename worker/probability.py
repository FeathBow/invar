import hashlib
import json
from dataclasses import dataclass

import torch

FORMAT = "invar-probabilities-v1"
ROLES = ("behavior", "proximal", "reference", "current", "advantage")
ENCODINGS = {torch.float32: ("F32", torch.int32, (1 << 32) - 1),
             torch.float64: ("F64", torch.int64, (1 << 64) - 1)}


@dataclass(frozen=True, kw_only=True)
class Observation:
    sample: str
    dtype: str
    words: tuple[tuple[int, ...], ...]
    active: tuple[bool, ...]


def words(tensor):
    _, integer, mask = ENCODINGS[tensor.dtype]
    return tuple(value & mask for value in tensor.detach().contiguous().view(integer).cpu().tolist())


def capture(sample, tokens):
    dtype, _, _ = ENCODINGS[tokens.current.dtype]
    return Observation(sample=sample, dtype=dtype,
                       words=tuple(words(getattr(tokens, role)) for role in ROLES),
                       active=tuple(tokens.active.detach().cpu().tolist()))


def document(observations, invocation, request):
    samples = [{"sample": item.sample, "dtype": item.dtype,
                **{role: list(value) for role, value in zip(ROLES, item.words, strict=True)},
                "active": list(item.active)} for item in observations]
    return {"format": FORMAT, "invocation": invocation, "request": request, "samples": samples}


def save(path, observations, *, invocation, request):
    encoded = json.dumps(document(observations, invocation, request), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    with path.open("xb") as output:
        output.write(encoded)
    return hashlib.sha256(encoded).hexdigest()
