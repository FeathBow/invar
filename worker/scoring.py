from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import math
from pathlib import Path

from worker import core, scalar
from worker.cohort import fields, identity
from worker.distribution import Probe
from worker.hf.session import decode as decode_call
from worker.invocation import request as request_value
from worker.trajectory import Request


@dataclass(frozen=True, kw_only=True)
class TokenPath:
    prefix: tuple[int, ...]
    response: tuple[int, ...]

    def __post_init__(self):
        for part in (self.prefix, self.response):
            if not isinstance(part, tuple) or not part or any(type(token) is not int or token < 0 for token in part):
                raise ValueError("A scoring path requires nonempty immutable token sequences")


@dataclass(frozen=True, kw_only=True)
class Source:
    path: TokenPath
    request: Request
    tokenizer: str
    truncated: bool
    inspection: bytes


def prescribed(request, path, tokenizer, prefix):
    if not isinstance(path, TokenPath):
        raise ValueError("Native scoring requires an immutable prescribed token path")
    if path.prefix != prefix:
        raise ValueError("Scoring source prefix differs from the actual target tokenizer")
    if any(token >= len(tokenizer) for token in (*path.prefix, *path.response)):
        raise ValueError("Scoring source has a token outside the target tokenizer vocabulary")
    eos = tokenizer.eos_token_id
    if eos in path.response[:-1] or len(path.response) > request.limit:
        raise ValueError("Scoring source extends past the declared stopping boundary")
    if path.response[-1] != eos and len(path.response) != request.limit:
        raise ValueError("Scoring source does not reach EOS or the declared horizon")


def source(encoded):
    value = core.decode(encoded)
    fields(value, "log_sha256 binding tokens behavior_bits prompt_length text truncated model revision adapter tokenizer base assembly request")
    for name in ("log_sha256", "adapter", "tokenizer", "base", "assembly"):
        identity(value[name])
    fields(value["binding"], "call attempt instance")
    if any(type(item) is not int or item < 0 for item in value["binding"].values()):
        raise ValueError("Scoring source requires nonnegative invocation identities")
    requested = fields(value["request"], "prompt tokens temperature seed")
    if not isinstance(requested["prompt"], str) or "\0" in requested["prompt"] or type(requested["seed"]) is not int:
        raise ValueError("Scoring source has an invalid prompt or seed")
    if type(requested["tokens"]) is not int or requested["tokens"] <= 0:
        raise ValueError("Scoring source requires a positive horizon")
    thermal = requested["temperature"]
    if type(thermal) not in (int, float) or not math.isfinite(thermal) or thermal <= 0:
        raise ValueError("Scoring source requires a finite positive temperature")
    if not isinstance(value["tokens"], list) or type(value["prompt_length"]) is not int:
        raise ValueError("Scoring source requires a token array and prompt boundary")
    boundary = value["prompt_length"]
    if not 0 < boundary < len(value["tokens"]):
        raise ValueError("Scoring source has an invalid prompt boundary")
    path = TokenPath(prefix=tuple(value["tokens"][:boundary]), response=tuple(value["tokens"][boundary:]))
    bits = value["behavior_bits"]
    if not isinstance(bits, list) or len(bits) != len(path.response):
        raise ValueError("Scoring source is missing sampled behavior words")
    for word in bits:
        scalar.probability(word)
    if type(value["truncated"]) is not bool or len(path.response) > requested["tokens"]:
        raise ValueError("Scoring source has an invalid stopping boundary")
    if value["truncated"] and len(path.response) != requested["tokens"]:
        raise ValueError("Truncated scoring source does not reach its horizon")
    if any(not isinstance(value[name], str) or not value[name] for name in ("model", "revision")) or not isinstance(value["text"], str):
        raise ValueError("Scoring source has an invalid model description or response")
    request = Request(sample="path", group="path", prompt=requested["prompt"], seed=requested["seed"],
                      limit=requested["tokens"], temperature=thermal)
    return Source(path=path, request=request, tokenizer=value["tokenizer"],
                  truncated=value["truncated"], inspection=bytes(encoded))


def decode(value):
    full = "probe_steps" in value
    fields(value, "binding program load adapter tokenizer base assembly request source_inspection" + (" probe_steps" if full else ""))
    encoded = value["source_inspection"]
    if not isinstance(encoded, str):
        raise ValueError("The checked score call must carry the exact source inspection text")
    selected = source(encoded.encode("utf-8"))
    probe = None
    if full:
        steps = value["probe_steps"]
        if not isinstance(steps, list):
            raise ValueError("Probe steps must be a JSON array")
        probe = Probe(steps=tuple(steps))
        probe.validate(len(selected.path.response))
    call = decode_call({key: item for key, item in value.items() if key not in ("source_inspection", "probe_steps")})
    if request_value(selected.request) != request_value(call.request) or selected.tokenizer != call.identities["tokenizer"]:
        raise ValueError("Score source differs from the declared target request or tokenizer")
    return call, selected, probe


def provenance(modules, *, packages):
    files = {}
    for name in modules:
        spec = find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError("Scoring implementation source is unavailable: " + name)
        files[name] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    return {"sources_sha256": files, "packages": {name: version(name) for name in packages}}
