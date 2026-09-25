import math
import re
from decimal import Decimal

import torch

from worker.hf.decoding import forward
from worker.tokenization import decode, prompt
from worker.trajectory import Request, Trajectory

ANSWER = re.compile(r"#### ([+-]?[0-9]+(?:\.[0-9]+)?)")


def generate(model, tokenizer, request, *, device="cuda"):
    if request.limit <= 0 or not math.isfinite(request.temperature) or request.temperature <= 0:
        raise ValueError("Generation requires a positive token limit and finite positive temperature")
    tokens = prompt(tokenizer, request.prompt).to(device)
    prompt_length = tokens.shape[1]
    generator = torch.Generator(device=device).manual_seed(request.seed)
    observations = []
    stopped = False
    cache = None
    with torch.no_grad():
        for _ in range(request.limit):
            logits, cache = forward(model, tokens, cache=cache)
            weights = torch.softmax(logits.float() / request.temperature, dim=-1)
            if not weights.isfinite().all() or weights.sum() <= 0:
                raise RuntimeError("Non-finite or empty behavior distribution")
            chosen = torch.multinomial(weights, 1, generator=generator)
            logprob = (weights.gather(-1, chosen) / weights.sum(-1, keepdim=True)).log()
            if not logprob.isfinite().all():
                raise RuntimeError("Invalid selected behavior probability")
            observations.append(logprob.reshape(()).cpu())
            tokens = torch.cat((tokens, chosen), dim=1)
            stopped = chosen.item() == tokenizer.eos_token_id
            if stopped:
                break
    return Trajectory(request=request, tokens=tokens.cpu(), prompt_length=prompt_length,
                      behavior=torch.stack(observations), truncated=not stopped,
                      text=decode(tokenizer, tokens[0, prompt_length:]))


def logprobs(model, trajectory, *, device="cuda"):
    tokens = trajectory.tokens.to(device)
    response = tokens[:, trajectory.prompt_length:]
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False,
                   logits_to_keep=response.shape[1] + 1).logits[:, :-1, :].float()
    selected = torch.log_softmax(logits / trajectory.request.temperature, dim=-1).gather(-1, response.unsqueeze(-1)).reshape(-1)
    if not selected.isfinite().all():
        raise RuntimeError("Non-finite selected model log probability")
    return selected


def reward(text, answer, truncated):
    expected = ANSWER.fullmatch(answer.strip())
    if expected is None:
        raise ValueError("Invalid frozen expected-answer grammar")
    if truncated or not text.strip():
        return 0.0
    actual = ANSWER.fullmatch(text.strip().splitlines()[-1])
    return float(actual is not None and Decimal(actual[1]) == Decimal(expected[1]))
