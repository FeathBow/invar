from dataclasses import dataclass
import math
import struct

import torch

from worker.distribution import Observed, Probe
from worker.probepacked import Packed
from worker.scoring import TokenPath


@dataclass(frozen=True, kw_only=True)
class Scored:
    path: TokenPath
    log_probability_bits: tuple[int, ...]
    truncated: bool
    native_sample_rows: int
    ignored_prefill_rows: int
    probe: Observed | Packed | None = None


def probability_words(tokens, observations):
    if observations is None or len(observations) != len(tokens):
        raise ValueError("Native scoring requires one probability per prescribed token")
    values = []
    for token, observed in zip(tokens, observations, strict=True):
        if token not in observed:
            raise ValueError("Native score omitted a prescribed token probability")
        value = observed[token].logprob
        if type(value) is not float or math.isnan(value) or value > 0:
            raise ValueError("Native score requires a nonpositive FP32 reported log or negative infinity")
        values.append(value)
    actual = torch.tensor(values, dtype=torch.float32)
    if actual.tolist() != values:
        raise ValueError("Native scores are not exact FP32 values")
    return words(values)


def words(values):
    return tuple(struct.unpack("=I", struct.pack("=f", value))[0] for value in values)


def trace_probe(trace, expected):
    if expected is None:
        if "probe" in trace:
            raise ValueError("Native scoring returned an undeclared mass observation")
        return None
    result = trace.get("probe")
    if not isinstance(result, (Observed, Packed)) or result.probe != expected:
        raise ValueError("Native mass observation differs from the declared response steps")
    if isinstance(result, Packed):
        result.validate()
    return result


def trace_score(trace, path, delivered, *, probe=None):
    probabilities, capped = delivered
    if tuple(trace["tokens"]) != path.response or words(trace["behavior"]) != probabilities:
        raise ValueError("Native output differs from observed prescribed tokens or probability words")
    total, ignored = trace["native_sample_rows"], trace["ignored_prefill_rows"]
    if type(total) is not int or type(ignored) is not int or ignored < 0 or total - ignored != len(path.response):
        raise ValueError("Native scoring has inconsistent sampled-row accounting")
    return Scored(path=path, log_probability_bits=probabilities, truncated=capped,
                  native_sample_rows=total, ignored_prefill_rows=ignored, probe=trace_probe(trace, probe))


def worker_scores(worker, queued, *, paths, delivered, probes):
    if set(worker["requests"]) != {bound.internal for bound in queued} or not worker["steps"]:
        raise ValueError("Native scoring observations omit owned requests or model steps")
    return tuple(trace_score(worker["requests"][bound.internal], path, result, probe=probe)
                 for bound, path, result, probe in zip(queued, paths, delivered, probes, strict=True))


def observed(delivered, queued, *, paths, observations, probes):
    if not observations:
        raise ValueError("Native scoring returned no worker observations")
    records = tuple(worker_scores(worker, queued, paths=paths, delivered=delivered, probes=probes) for worker in observations)
    if any(record != records[0] for record in records[1:]):
        raise ValueError("Native workers returned different prescribed-path observations")
    return records[0]


def probe_selection(paths, probes):
    selected = (None,) * len(paths) if probes is None else tuple(probes)
    if len(selected) != len(paths):
        raise ValueError("Native scoring requires one optional probe declaration per path")
    for path, probe in zip(paths, selected, strict=True):
        if probe is not None:
            if not isinstance(probe, Probe):
                raise ValueError("Native scoring requires immutable probe declarations")
            probe.validate(len(path.response))
    return selected
