import hashlib
import json

import torch

HASH_CHUNK_BYTES = 4 * 1024 * 1024
METADATA_LENGTH_BYTES = 8
FORMAT = b"invar-frozen-state-v1\0"


def blocks(value):
    if value.numel() * value.element_size() <= HASH_CHUNK_BYTES:
        yield value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return
    axis = next(index for index, size in enumerate(value.shape) if size > 1)
    for part in value.split(max(1, value.shape[axis] // 2), dim=axis):
        yield from blocks(part)


def digest(model):
    trainable = {name for name, value in model.named_parameters(remove_duplicate=False) if value.requires_grad}
    result = hashlib.sha256(FORMAT)
    for name, value in sorted(model.state_dict().items()):
        if name in trainable:
            continue
        metadata = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        result.update(len(metadata).to_bytes(METADATA_LENGTH_BYTES, "big"))
        result.update(metadata)
        for block in blocks(value):
            result.update(block)
    return result.hexdigest()


def verify(model, expected):
    if not isinstance(expected, str) or expected != digest(model):
        raise RuntimeError("Checkpoint frozen base binding mismatch")
