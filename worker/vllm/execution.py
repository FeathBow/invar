from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json

import torch

from worker.hf.tensors import assert_equal
from worker.tokenization import prompt
from worker.vllm.state import Binding
from worker.vllm.lora import Consumption
from worker.vllm.mapping import selection
from worker.vllm.profile import Loaded
from worker.vllm.rollout import LOGPROBS_MODE, lora_selection, parameters, trajectory
from worker.vllm.worker import begin, end, permit


@dataclass(frozen=True, kw_only=True)
class Prepared:
    bindings: tuple[Binding, ...]
    resident: tuple[tuple[Consumption, ...], ...]
    models: tuple[tuple[Loaded, ...], ...]


@dataclass(frozen=True, kw_only=True)
class Execution:
    prepared: Prepared
    trajectories: tuple
    observations: tuple[dict, ...]


def enqueued(engine, identities, *, requests, prefixes, choices, receipts, eos):
    states = engine.llm_engine.output_processor.request_states
    if len(set(identities)) != len(requests) or set(states) != set(identities):
        raise ValueError("Native enqueue returned a different or repeated owned request set")
    result = tuple(Binding(internal=identity, external=states[identity].external_req_id, request=request,
                           prompt=prefix, eos=eos, selection=choice, receipt=receipt)
                   for identity, request, prefix, choice, receipt in
                   zip(identities, requests, prefixes, choices, receipts, strict=True))
    if len({value.external for value in result}) != len(result):
        raise ValueError("Native output identities alias different owned requests")
    return result


def bindings(engine, requests, *, prefixes, loras, receipts, eos):
    if engine.llm_engine.has_unfinished_requests():
        raise RuntimeError("Native execution requires an engine with no other queued requests")
    if engine.llm_engine.model_config.logprobs_mode != LOGPROBS_MODE:
        raise ValueError("Native execution requires processed behavior log probabilities")
    selected = tuple(selection(lora) for lora in loras)
    identities = engine.enqueue([{"prompt_token_ids": list(prefix)} for prefix in prefixes],
                                sampling_params=[parameters(request, eos=eos) for request in requests],
                                lora_request=list(loras), use_tqdm=False)
    try:
        return enqueued(engine, identities, requests=requests, prefixes=prefixes, choices=selected, receipts=receipts, eos=eos)
    except BaseException:
        engine.llm_engine.abort_request(identities, internal=True)
        raise


@contextmanager
def owned(engine, queued):
    started = False
    completed = False
    try:
        workers = engine.collective_rpc(begin, kwargs={"bindings": [asdict(value) for value in queued]})
        started = True
        observations = []
        prepared = Prepared(bindings=queued,
                            resident=tuple(tuple(Consumption(**value) for value in worker["resident"]) for worker in workers),
                            models=tuple(tuple(Loaded(**value) for value in worker["models"]) for worker in workers))
        yield prepared, observations
        completed = True
    finally:
        try:
            if started:
                observations.extend(engine.collective_rpc(end, kwargs={"completed": completed}))
        finally:
            if not completed:
                engine.llm_engine.abort_request([value.internal for value in queued], internal=True)


def outputs(values, queued):
    result = {value.request_id: value for value in values}
    if len(result) != len(values) or set(result) != {value.external for value in queued}:
        raise ValueError("Native completed outputs differ from the owned request set")
    return result


def delivered(output, tokenizer, bound):
    from vllm.lora.request import LoRARequest

    expected_lora = LoRARequest(**json.loads(bound.selection.description))
    if lora_selection(output.lora_request) != bound.selection.description:
        raise ValueError("Native completed output changed the approved LoRA selection")
    return trajectory(output, tokenizer, bound.request, prefix=bound.prompt, lora=expected_lora)


def compared(trajectories, queued, observations):
    if not observations:
        raise ValueError("Native execution returned no worker observations")
    for observed in observations:
        if set(observed["requests"]) != {value.internal for value in queued} or not observed["steps"]:
            raise ValueError("Native execution observations do not cover the owned request set")
        for actual, bound in zip(trajectories, queued, strict=True):
            trace = observed["requests"][bound.internal]
            assert_equal(actual.tokens[0, actual.prompt_length:], torch.tensor(trace["tokens"], dtype=torch.int64))
            assert_equal(actual.behavior, torch.tensor(trace["behavior"], dtype=torch.float32))


def generate(engine, tokenizer, requests, *, loras, receipts, approve):
    requests, loras, receipts = tuple(requests), tuple(loras), tuple(receipts)
    if not requests or len(requests) != len(loras) or len(requests) != len(receipts):
        raise ValueError("Native execution requires a prepared package for every request")
    prefixes = tuple(tuple(prompt(tokenizer, request.prompt)[0].tolist()) for request in requests)
    queued = bindings(engine, requests, prefixes=prefixes, loras=loras, receipts=receipts, eos=tokenizer.eos_token_id)
    with owned(engine, queued) as (prepared, observations):
        approve(prepared)
        engine.collective_rpc(permit)
        actual = outputs(engine.wait_for_completion(use_tqdm=False), queued)
        result = tuple(delivered(actual[bound.external], tokenizer, bound) for bound in queued)
    compared(result, queued, observations)
    return Execution(prepared=prepared, trajectories=result, observations=tuple(observations))
