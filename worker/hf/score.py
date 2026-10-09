import torch

from worker import float32
from worker.hf.decoding import forward
from worker.hf.model import adapter_state
from worker.hf.policy import activate, read_adapter
from worker.hf.rollout import distribution, selected
from worker.implementation import INFERENCE


def score(model, trajectory, *, device):
    tokens = trajectory.tokens.to(device)
    observations = []
    cache = None
    with torch.no_grad():
        for position in range(trajectory.prompt_length, tokens.shape[1]):
            logits, cache = forward(model, tokens[:, :position], cache=cache)
            observations.append(selected(distribution(logits, trajectory.request.temperature), tokens[:, position:position + 1]).cpu())
    return torch.stack(observations)


def referenced(runtime, trajectories, reference, *, identities):
    if reference is None:
        return (None,) * len(trajectories)
    policy = adapter_state(runtime.model)
    materialization = {"base": identities["base"], "assembly": identities["assembly"], "role": INFERENCE}
    try:
        activate(runtime.model, read_adapter(reference.adapter, reference.digest), **materialization)
        scores = tuple(score(runtime.model, trajectory, device=runtime.device) for trajectory in trajectories)
    finally:
        restored = activate(runtime.model, policy, **materialization)
    if restored != identities["adapter"]:
        raise RuntimeError("Reference scoring did not restore the sampled policy")
    return tuple((reference.digest, tuple(float32.word(value) for value in observed.tolist())) for observed in scores)
