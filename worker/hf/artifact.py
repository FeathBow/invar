import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from safetensors.torch import load as load_tensors

from worker.core import decode
from worker.hf.policy import validate_identity, validate_schema, validate_state
from worker.hf.tensors import digest

FORMAT = "invar-peft-handoff-v1"
CHECKPOINT_FORMAT = "invar-peft-checkpoint-input-v1"
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
RECEIPT = "handoff.json"
SOURCE_FIELDS = frozenset(("adapter", "base", "assembly", "tokenizer"))


@dataclass(frozen=True, kw_only=True)
class Package:
    identity: str
    source: Mapping[str, str]
    configuration: str
    tensors: Mapping[str, torch.Tensor]

    def verify(self):
        validate_state(self.tensors, self.source["adapter"])


def checked_bytes(path, expected):
    validate_identity(expected)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"PEFT handoff file binding mismatch: {path.name}")
    return content


def read(directory, *, expected):
    receipt = decode(checked_bytes(directory / RECEIPT, expected))
    if not isinstance(receipt, dict) or receipt.get("format") != FORMAT:
        raise ValueError("Unsupported PEFT handoff format")
    source = receipt.get("source")
    files = receipt.get("files")
    if not isinstance(source, dict) or source.keys() != SOURCE_FIELDS:
        raise ValueError("PEFT handoff requires all four source bindings")
    if not isinstance(files, dict) or files.keys() != {CONFIG, WEIGHTS}:
        raise ValueError("PEFT handoff requires configuration and tensor file bindings")
    for identity in (*source.values(), receipt.get("producer")):
        validate_identity(identity)
    configuration = checked_bytes(directory / CONFIG, files[CONFIG]).decode("utf-8")
    if not isinstance(decode(configuration), dict):
        raise ValueError("PEFT configuration must be an object")
    # Decode the same bytes that were hashed; do not reopen a mutable path.
    tensors = load_tensors(checked_bytes(directory / WEIGHTS, files[WEIGHTS]))
    package = Package(identity=expected, source=MappingProxyType(source),
                      configuration=configuration, tensors=MappingProxyType(tensors))
    package.verify()
    return package


def read_checkpoint(path, *, template, expected):
    original = read(template, expected=expected)
    tensors = load_tensors(path.read_bytes())
    validate_schema(original.tensors, tensors)
    policy = digest(tensors)
    validate_state(tensors, policy)
    binding = json.dumps({"format": CHECKPOINT_FORMAT, "template": original.identity, "policy": policy},
                         sort_keys=True, separators=(",", ":"))
    return Package(identity=hashlib.sha256(binding.encode()).hexdigest(),
                   source=MappingProxyType({**original.source, "adapter": policy}),
                   configuration=original.configuration, tensors=MappingProxyType(tensors))
