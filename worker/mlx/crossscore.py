from dataclasses import dataclass

import mlx.core as mx

from worker.distribution import Observed, Probe
from worker.mlx.distribution import Capture
from worker.mlx.words import words
from worker.mlx.rollout import Sampler, consumed, run
from worker.trajectory import Request
from worker.scoring import TokenPath, bounded
from worker.probestore import Stored


@dataclass(frozen=True, kw_only=True)
class ScoredPath:
    request: Request
    path: TokenPath
    log_probability_bits: tuple[int, ...]
    lookahead_draws: int
    probe: Observed | Stored | None = None


class PathSampler(Sampler):
    def __init__(self, request, path, *, width, probe=None, store=None):
        super().__init__(request)
        self.path = path
        self.width = width
        self.position = 0
        if probe is None and store is not None:
            raise ValueError("Native scoring storage requires a declared probe")
        self.capture = None if probe is None else Capture(probe, store=store)
        if self.capture is not None:
            self.capture.probe.validate(len(path.response))

    def process_logits(self, tokens, logits):
        expected = (*self.path.prefix, *self.path.response[:self.position])
        if tokens.ndim != 1 or tuple(tokens.tolist()) != expected:
            raise ValueError("Native scoring context differs from the prescribed path")
        if logits.shape[-1] != self.width:
            raise ValueError("Native distribution width differs from the target's output coordinates")
        return super().process_logits(tokens, logits)

    def select(self, distribution):
        position = self.position
        if self.capture is not None:
            self.capture.observe(position, distribution)
        drawn = super().select(distribution)
        self.position += 1
        if position < len(self.path.response):
            return mx.array([self.path.response[position]], dtype=drawn.dtype)
        if position == len(self.path.response):
            return drawn
        raise RuntimeError("Native scoring advanced beyond its single unused lookahead")

    def consume(self, token):
        return consumed(self.pending, token, zero_support=True)

    def completed(self):
        if self.position != len(self.path.response) + 1 or len(self.pending) != 1:
            raise RuntimeError("Native scoring did not consume exactly its prescribed path and one lookahead")
        return None if self.capture is None else self.capture.completed()


def coordinates(model):
    language = model.language_model
    rows = language.model.embed_tokens.weight.shape[0]
    return {"input": rows, "output": rows if language.args.tie_word_embeddings else language.lm_head.weight.shape[0]}


def capture_selections(count, *, probes, stores):
    selected = (None,) * count if probes is None else tuple(probes)
    if len(selected) != count or any(probe is not None and not isinstance(probe, Probe) for probe in selected):
        raise ValueError("Cached scoring requires one valid probe selection per request")
    retained = (None,) * count if stores is None else tuple(stores)
    if len(retained) != count:
        raise ValueError("Cached scoring requires one optional probe store per request")
    return selected, retained


def score(model, requests, *, paths, coordinated, sampling, probes=None, stores=None):
    requests, paths = tuple(requests), tuple(paths)
    if not requests or len(requests) != len(paths):
        raise ValueError("Cached scoring requires one complete path per request")
    selected, retained = capture_selections(len(requests), probes=probes, stores=stores)
    for request, path in zip(requests, paths, strict=True):
        bounded(request, path, coordinated)
        if path.response[-1] >= coordinated["input"]:
            raise ValueError("Native scoring feeds the last response token to its lookahead step, outside the input embedding")
    samplers = [PathSampler(request, path, width=coordinated["output"], probe=probe, store=store)
                for request, path, probe, store in zip(requests, paths, selected, retained, strict=True)]
    produced = run(model, [path.prefix for path in paths], limits=[len(path.response) for path in paths], stop=(),
                   sampling=sampling, samplers=samplers)
    result = []
    for (tokens, behavior, reason), sampler, request, path in zip(produced, samplers, requests, paths, strict=True):
        observed_probe = sampler.completed()
        if tuple(tokens) != path.response or reason != "length":
            raise RuntimeError("Native cached scoring returned a different token path")
        result.append(ScoredPath(request=request, path=path, log_probability_bits=words(mx.stack(behavior)),
                                 lookahead_draws=1, probe=observed_probe))
    return tuple(result)
