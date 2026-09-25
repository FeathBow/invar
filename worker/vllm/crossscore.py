from dataclasses import dataclass

from worker.scoring import TokenPath, prescribed
from worker.tokenization import prompt
from worker.vllm import execution, rollout
from worker.vllm.scoreobservation import Scored, observed, probability_words, probe_selection, words
from worker.vllm.worker import permit


@dataclass(frozen=True, kw_only=True)
class Execution:
    prepared: execution.Prepared
    scores: tuple[Scored, ...]
    observations: tuple[dict, ...]


def delivered(output, tokenizer, *, bound, path, vocabulary):
    from vllm.lora.request import LoRARequest
    import json

    lora = LoRARequest(**json.loads(bound.selection.description))
    result = rollout.completion(output, prefix=path.prefix, lora=lora)
    tokens = rollout.response_tokens(result, limit=bound.request.limit, vocabulary=vocabulary)
    if tokens != path.response:
        raise ValueError("Native output differs from the complete prescribed response")
    capped = rollout.truncated(result, tokens=tokens, eos=bound.eos, limit=bound.request.limit)
    return probability_words(tokens, result.logprobs), capped


def score(engine, tokenizer, requests, *, paths, loras, receipts, approve, measure=None, probes=None):
    requests, paths, loras, receipts = tuple(requests), tuple(paths), tuple(loras), tuple(receipts)
    if not requests or len({len(requests), len(paths), len(loras), len(receipts)}) != 1:
        raise ValueError("Native scoring requires one path and prepared adapter per request")
    for request, path in zip(requests, paths, strict=True):
        prescribed(request, path, tokenizer, tuple(prompt(tokenizer, request.prompt)[0].tolist()))
    selected_probes = probe_selection(paths, probes)
    queued = execution.bindings(engine, requests, prefixes=tuple(path.prefix for path in paths),
                                 loras=loras, receipts=receipts, eos=tokenizer.eos_token_id)
    responses = {bound.internal: path.response for bound, path in zip(queued, paths, strict=True)}
    declared = {bound.internal: probe for bound, probe in zip(queued, selected_probes, strict=True) if probe is not None}
    with execution.owned(engine, queued, paths=responses, probes=declared or None) as (prepared, observations):
        approve(prepared)
        engine.collective_rpc(permit)
        def complete():
            return engine.wait_for_completion(use_tqdm=False)
        completed = complete() if measure is None else measure("cross_score", complete)
        outputs = execution.outputs(completed, queued)
        vocabulary = len(tokenizer)
        results = tuple(delivered(outputs[bound.external], tokenizer, bound=bound, path=path, vocabulary=vocabulary)
                        for bound, path in zip(queued, paths, strict=True))
    scores = observed(results, queued, paths=paths, observations=observations, probes=selected_probes)
    return Execution(prepared=prepared, scores=scores, observations=tuple(observations))
