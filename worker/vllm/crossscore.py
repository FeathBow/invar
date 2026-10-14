from dataclasses import dataclass

from worker.scoring import TokenPath, bounded
from worker.vllm import execution, rollout
from worker.vllm.scoreobservation import Scored, observed, probability_words, probe_selection, words
from worker.vllm.worker import permit


@dataclass(frozen=True, kw_only=True)
class Execution:
    prepared: execution.Prepared
    scores: tuple[Scored, ...]
    observations: tuple[dict, ...]


def delivered(output, *, bound, path, vocabulary):
    from vllm.lora.request import LoRARequest
    import json

    description = bound.selection.description
    lora = None if description is None else LoRARequest(**json.loads(description))
    result = rollout.completion(output, prefix=path.prefix, lora=lora)
    tokens = rollout.response_tokens(result, limit=len(path.response), vocabulary=vocabulary)
    if tokens != path.response or (result.finish_reason, result.stop_reason) != ("length", None):
        raise ValueError("Native output differs from the complete prescribed response")
    return probability_words(tokens, result.logprobs)


def coordinates(engine):
    width = engine.llm_engine.model_config.get_vocab_size()
    return {"input": width, "output": width}


def score(engine, requests, *, paths, loras, receipts, approve, coordinated, measure=None, probes=None):
    requests, paths, loras, receipts = tuple(requests), tuple(paths), tuple(loras), tuple(receipts)
    if not requests or len({len(requests), len(paths), len(loras), len(receipts)}) != 1:
        raise ValueError("Native scoring requires one path and prepared adapter per request")
    for request, path in zip(requests, paths, strict=True):
        bounded(request, path, coordinated)
    selected_probes = probe_selection(paths, probes)
    queued = execution.bindings(engine, requests, prefixes=tuple(path.prefix for path in paths), loras=loras, receipts=receipts,
                                 limits=tuple(len(path.response) for path in paths), stop=())
    responses = {bound.internal: path.response for bound, path in zip(queued, paths, strict=True)}
    declared = {bound.internal: probe for bound, probe in zip(queued, selected_probes, strict=True) if probe is not None}
    with execution.owned(engine, queued, paths=responses, probes=declared or None) as (prepared, observations):
        approve(prepared)
        engine.collective_rpc(permit)
        def complete():
            return engine.wait_for_completion(use_tqdm=False)
        completed = complete() if measure is None else measure("cross_score", complete)
        outputs = execution.outputs(completed, queued)
        results = tuple(delivered(outputs[bound.external], bound=bound, path=path, vocabulary=coordinated["output"])
                        for bound, path in zip(queued, paths, strict=True))
    scores = observed(results, queued, paths=paths, observations=observations, probes=selected_probes)
    return Execution(prepared=prepared, scores=scores, observations=tuple(observations))
