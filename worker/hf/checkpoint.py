from dataclasses import asdict, dataclass

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from worker.hf import assembly
from worker import binding
from worker import implementation
from worker.hf import frozen
from worker.hf import operation
from worker.hf.model import adapter_state
from worker.hf.tensors import assert_equal, digest

LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8


@dataclass(frozen=True, kw_only=True)
class CheckpointIdentity:
    adapter: str
    base: str
    assembly: str
    tokenizer: str


def checkpoint_state(model, optimizer, *, tokenizer):
    parameters = binding.parameters(model, optimizer)
    tokenizer_digest = operation.digest(tokenizer)
    configuration = assembly.digest(model, implementation.LEARNING)
    base = frozen.digest(model)
    state = adapter_state(model)
    actual = CheckpointIdentity(adapter=digest(state), base=base, assembly=configuration,
                                tokenizer=tokenizer_digest)
    training = {**asdict(actual),
                "parameters": parameters,
                "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}
    return state, training


def checkpoint(model, optimizer, output, *, tokenizer, expected):
    state, training = checkpoint_state(model, optimizer, tokenizer=tokenizer)
    actual = CheckpointIdentity(**{key: training[key] for key in ("adapter", "base", "assembly", "tokenizer")})
    if expected is not None and actual != expected:
        raise RuntimeError("Checkpoint identity differs from the declared successor")
    save_file(state, output / "adapter.safetensors")
    torch.save(training, output / "learner.pt")
    assert_equal(state, load_file(output / "adapter.safetensors"))
    assert_equal(training, torch.load(output / "learner.pt", weights_only=True))
    return state


def restore(model, optimizer, output, *, tokenizer):
    state = load_file(output / "adapter.safetensors")
    training = torch.load(output / "learner.pt", weights_only=True)
    restore_state(model, optimizer, (state, training), tokenizer=tokenizer)


def restore_state(model, optimizer, snapshot, *, tokenizer):
    state, training = snapshot
    if training["adapter"] != digest(state):
        raise RuntimeError("Checkpoint adapter binding mismatch")
    operation.verify(tokenizer, training.get("tokenizer"))
    assembly.verify(model, training.get("assembly"), implementation.LEARNING)
    frozen.verify(model, training.get("base"))
    training = binding.restore(training, model, optimizer)
    set_peft_model_state_dict(model, state)
    assert_equal(state, adapter_state(model))
    optimizer.load_state_dict(training["optimizer"])
    torch.set_rng_state(training["cpu_rng"])
    torch.cuda.set_rng_state_all(training["cuda_rng"])
    assert_equal(training["optimizer"], optimizer.state_dict())
