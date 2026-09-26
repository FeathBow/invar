from collections import deque
from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import BatchGenerator

from worker.mlx.projection import SPLIT_ROWS, RowLinear
from worker.mlx.temperature import tempered
from worker.mlx.tokenization import prompt
from worker.tokenization import decode
from worker.trajectory import Trajectory


@dataclass(frozen=True, kw_only=True)
class Sampling:
    batch_size: int
    prefill_step: int

    def __post_init__(self):
        if self.batch_size <= 0 or self.prefill_step <= 0:
            raise ValueError("Native generation requires positive batch and prefill sizes")


class Sampler:
    def __init__(self, request):
        if not math.isfinite(request.temperature) or request.temperature <= 0 or request.limit <= 0:
            raise ValueError("Generation requires positive temperature and token limit")
        self.temperature = request.temperature
        self.key = mx.random.key(request.seed)
        self.pending = deque()

    def process_logits(self, tokens, logits):
        return fp32_logits(tokens, logits)

    def __call__(self, logprobs):
        if logprobs.dtype != mx.float32 or logprobs.ndim != 2 or logprobs.shape[0] != 1:
            raise ValueError("The request sampler requires one FP32 native distribution")
        distribution, weights, total = tempered(logprobs, self.temperature)
        chosen = self.select(distribution)
        selected = mx.log(mx.take_along_axis(distribution, chosen[:, None], axis=-1)).reshape(())
        finite = mx.all(mx.isfinite(weights))
        mx.async_eval(chosen, selected, finite, total)
        self.pending.append((chosen, selected, finite, total))
        return chosen

    def select(self, distribution):
        self.key, draw = mx.random.split(self.key)
        return mx.random.categorical(mx.log(distribution), axis=-1, key=draw)

    def consume(self, token):
        return consumed(self.pending, token, zero_support=False)


def consumed(pending, token, *, zero_support):
    if not pending:
        raise RuntimeError("Native generator returned an unobserved sample")
    chosen, selected, finite, total = pending.popleft()
    mx.eval(chosen, selected, finite, total)
    probability = selected.item()
    valid = math.isfinite(probability) or (zero_support and probability == -math.inf)
    if chosen.item() != token or not finite.item() or total.item() <= 0 or not valid:
        raise RuntimeError("Native token differs from its actual sampling distribution")
    return selected


def fp32_logits(_tokens, logits):
    return logits.astype(mx.float32)


def prefill_batch_size(model, sampling):
    projection = next(module for _, module in model.named_modules() if isinstance(module, nn.QuantizedLinear))
    if type(projection) is not RowLinear:
        return sampling.batch_size
    if sampling.batch_size >= SPLIT_ROWS:
        raise ValueError("Row-independent generation requires a decode batch below its stock projection rows")
    return 1


def generate(model, tokenizer, requests, *, sampling):
    return execute(model, tokenizer, requests, sampling=sampling, samplers=[Sampler(request) for request in requests])


def execute(model, tokenizer, requests, *, sampling, samplers):
    if not requests:
        raise ValueError("Native generation requires a nonempty request group")
    if len(samplers) != len(requests):
        raise ValueError("Native decoding requires one sampler per request")
    model.eval()
    prefixes = [prompt(tokenizer, request.prompt)[0].tolist() for request in requests]
    engine = BatchGenerator(model, completion_batch_size=sampling.batch_size,
                            prefill_batch_size=prefill_batch_size(model, sampling), prefill_step_size=sampling.prefill_step,
                            stop_tokens=[[tokenizer.eos_token_id]])
    try:
        uids = engine.insert(prefixes, max_tokens=[request.limit for request in requests], samplers=samplers,
                             logits_processors=[[sampler.process_logits] for sampler in samplers])
        return collect(engine, uids, (requests, prefixes, samplers), tokenizer=tokenizer)
    finally:
        engine.close()


def collect(engine, uids, inputs, *, tokenizer):
    requests, prefixes, samplers = inputs
    pending = {uid: (request, prefix, sampler, [], [])
               for uid, request, prefix, sampler in zip(uids, requests, prefixes, samplers, strict=True)}
    completed = {}
    while pending:
        responses = engine.next_generated()
        if not responses:
            raise RuntimeError("Native generation stopped before every requested trajectory completed")
        for response in responses:
            request, prefix, sampler, tokens, behavior = pending[response.uid]
            tokens.append(response.token)
            behavior.append(sampler.consume(response.token))
            if response.finish_reason is not None:
                stopped = response.token == tokenizer.eos_token_id
                if response.finish_reason != ("stop" if stopped else "length") or (not stopped and len(tokens) != request.limit):
                    raise RuntimeError("Native completion differs from the admitted EOS or token boundary")
                completed[response.uid] = Trajectory(request=request, tokens=mx.array([prefix + tokens], dtype=mx.int32),
                                                     prompt_length=len(prefix), behavior=mx.stack(behavior),
                                                     text=decode(tokenizer, tokens), truncated=not stopped)
                del pending[response.uid]
    return tuple(completed[uid] for uid in uids)

