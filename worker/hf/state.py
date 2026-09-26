from dataclasses import dataclass
import io

import torch

from worker import binding
from worker.hf.checkpoint import checkpoint_state
from worker.hf.tensors import fingerprint
from worker.update import snapshot


@dataclass(frozen=True, kw_only=True)
class Witness:
    policy: str
    learner: str
    signature: str


def observe(learner, tokenizer):
    _, actual = checkpoint_state(learner.model, learner.optimizer, tokenizer=tokenizer)
    return actual


def signature(training):
    return fingerprint(binding.observation(training))


def attest(learner, tokenizer, *, checkpoint, policy, expected):
    identity, encoded = snapshot(checkpoint / "learner.pt")
    if identity != expected:
        raise ValueError("Resident learner checkpoint differs from its declared file identity")
    saved = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    actual = observe(learner, tokenizer)
    if saved["adapter"] != policy or actual["adapter"] != policy or signature(saved) != signature(actual):
        raise RuntimeError("Resident learner state differs from the saved checkpoint")
    return Witness(policy=policy, learner=identity, signature=signature(saved))


def verify(learner, tokenizer, expected):
    actual = observe(learner, tokenizer)
    if signature(actual) != expected.signature:
        raise RuntimeError("Resident learner state differs from the saved checkpoint")
    return actual
