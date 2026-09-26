import json
import math

import torch

from worker.trajectory import Trajectory
from worker.tokenization import decode, prompt

LOGPROBS_MODE = "processed_logprobs"
LORA_FIELDS = ("lora_name", "lora_int_id", "lora_path", "base_model_name",
               "tensorizer_config_dict", "load_inplace", "is_3d_lora_weight")


def parameters(request, *, eos):
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    if type(request.limit) is not int or request.limit <= 0:
        raise ValueError("Generation requires a positive integer token limit")
    if not math.isfinite(request.temperature) or request.temperature <= 0:
        raise ValueError("Generation requires a finite positive temperature")
    if type(request.seed) is not int or type(eos) is not int or eos < 0:
        raise ValueError("Generation requires an integer seed and tokenizer EOS")
    result = SamplingParams(
        n=1, temperature=request.temperature, seed=request.seed,
        top_k=0, top_p=1.0, min_p=0.0,
        presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
        max_tokens=request.limit, min_tokens=0,
        # Own the exact EOS set instead of inheriting extra model stop tokens.
        ignore_eos=True, stop_token_ids=[eos], stop=[],
        logprobs=0, prompt_logprobs=None, detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    if result.temperature != request.temperature or result.seed != request.seed:
        raise ValueError("Native sampling parameters changed the requested temperature or seed")
    return result


def behavior(tokens, observations):
    if observations is None or len(observations) != len(tokens):
        raise ValueError("Native generation requires one probability observation per response token")
    values = []
    for token, observed in zip(tokens, observations, strict=True):
        if token not in observed:
            raise ValueError("Native probabilities omit the selected token")
        value = observed[token].logprob
        if type(value) is not float or not math.isfinite(value) or value > 0:
            raise ValueError("Native selected log probability must be finite and nonpositive")
        values.append(value)
    result = torch.tensor(values, dtype=torch.float32)
    if result.tolist() != values or not result.isfinite().all():
        raise ValueError("Native selected log probabilities are not exact FP32 values")
    return result


def lora_selection(value):
    if value is None:
        return None
    # Native LoRARequest equality compares only the name.
    return json.dumps({name: getattr(value, name) for name in LORA_FIELDS},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def completion(output, *, prefix, lora):
    if output.finished is not True or len(output.outputs) != 1:
        raise ValueError("Native generation requires one completed output per request")
    if tuple(output.prompt_token_ids or ()) != tuple(prefix) or lora_selection(output.lora_request) != lora_selection(lora):
        raise ValueError("Native output prompt or adapter differs from the requested input")
    return output.outputs[0]


def response_tokens(output, *, limit, vocabulary):
    tokens = tuple(output.token_ids)
    if output.index != 0 or not 0 < len(tokens) <= limit:
        raise ValueError("Native response index or token count differs from the request")
    if any(type(token) is not int or token < 0 or token >= vocabulary for token in tokens):
        raise ValueError("Native response contains an invalid tokenizer token ID")
    return tokens


def truncated(output, *, tokens, eos, limit):
    if eos in tokens[:-1]:
        raise ValueError("Native response continues after tokenizer EOS")
    ended = tokens[-1] == eos
    expected = ("stop", eos) if ended else ("length", None)
    if (output.finish_reason, output.stop_reason) != expected:
        raise ValueError("Native generation did not finish by tokenizer EOS or the requested token limit")
    if not ended and len(tokens) != limit:
        raise ValueError("Native generation reached a different token limit")
    return not ended


def trajectory(output, tokenizer, request, *, prefix, lora, vocabulary):
    result = completion(output, prefix=prefix, lora=lora)
    tokens = response_tokens(result, limit=request.limit, vocabulary=vocabulary)
    capped = truncated(result, tokens=tokens, eos=tokenizer.eos_token_id, limit=request.limit)
    return Trajectory(request=request, tokens=torch.tensor([[*prefix, *tokens]], dtype=torch.int64),
                      prompt_length=len(prefix), behavior=behavior(tokens, result.logprobs),
                      text=decode(tokenizer, tokens), truncated=capped)


def native_outputs(engine, *, prefixes, sampling, loras):
    outputs = engine.generate([{"prompt_token_ids": list(prefix)} for prefix in prefixes],
                              sampling_params=sampling, lora_request=list(loras), use_tqdm=False)
    if len(outputs) != len(prefixes) or len({output.request_id for output in outputs}) != len(outputs):
        raise ValueError("Native engine returned a different or repeated request set")
    return outputs


def generate(engine, tokenizer, requests, *, loras):
    requests, loras = tuple(requests), tuple(loras)
    if not requests or len(requests) != len(loras):
        raise ValueError("Native generation requires an adapter selection for every request")
    if engine.llm_engine.model_config.logprobs_mode != LOGPROBS_MODE:
        raise ValueError("Native generation requires processed behavior log probabilities")
    prefixes = tuple(tuple(prompt(tokenizer, request.prompt)[0].tolist()) for request in requests)
    sampling = [parameters(request, eos=tokenizer.eos_token_id) for request in requests]
    outputs = native_outputs(engine, prefixes=prefixes, sampling=sampling, loras=loras)
    vocabulary = len(tokenizer)
    return tuple(trajectory(output, tokenizer, request, prefix=prefix, lora=lora, vocabulary=vocabulary)
                 for output, request, prefix, lora in zip(outputs, requests, prefixes, loras, strict=True))
