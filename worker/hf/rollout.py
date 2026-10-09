import math

import torch

from worker.hf.decoding import forward
from worker.tokenization import decode, prompt
from worker.trajectory import Trajectory


def distribution(logits, temperature):
    weights = torch.softmax(logits.float() / temperature, dim=-1)
    if not weights.isfinite().all() or weights.sum() <= 0:
        raise RuntimeError("Non-finite or empty token distribution")
    return weights


def selected(weights, token):
    logprob = (weights.gather(-1, token) / weights.sum(-1, keepdim=True)).log()
    if not logprob.isfinite().all():
        raise RuntimeError("Invalid selected token probability")
    return logprob.reshape(())


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
            weights = distribution(logits, request.temperature)
            chosen = torch.multinomial(weights, 1, generator=generator)
            observations.append(selected(weights, chosen).cpu())
            tokens = torch.cat((tokens, chosen), dim=1)
            stopped = chosen.item() == tokenizer.eos_token_id
            if stopped:
                break
    return Trajectory(request=request, tokens=tokens.cpu(), prompt_length=prompt_length,
                      behavior=torch.stack(observations), truncated=not stopped,
                      text=decode(tokenizer, tokens[0, prompt_length:]))

