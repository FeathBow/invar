import hashlib
import json

import mlx.core as mx
from mlx.utils import tree_unflatten
import numpy as np

from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx.adapter import state as adapter, install
from worker.mlx import tensors as mlx_tensors
from worker import float32

KEY_SHAPE = (2,)
KEY_WORD_BITS = 32


def configuration(settings):
    return {"betas": list(settings.betas), "epsilon": settings.epsilon,
            "weight_decay": settings.weight_decay, "bias_correction": True}


def validate(snapshot, parameters, *, identities, settings):
    fields = {"format", "adapter", "base", "assembly", "tokenizer", "parameters", "optimizer", "rng"}
    if set(snapshot) != fields or snapshot["format"] != mlx_checkpoint.FORMAT:
        raise ValueError("Unexpected native learner checkpoint fields")
    if any(snapshot[name] != value for name, value in identities.items()):
        raise ValueError("Native learner checkpoint materialization differs from the requested input")
    if sorted(snapshot["parameters"]) != sorted(parameters):
        raise ValueError("Native optimizer parameter binding mismatch")
    optimizer = snapshot["optimizer"]
    if set(optimizer) != {"configuration", "state"} or optimizer["configuration"] != configuration(settings):
        raise ValueError("Native AdamW configuration differs from the requested input")
    state = optimizer["state"]
    step = state["step"]
    if step.dtype != mx.uint64 or step.shape != ():
        raise ValueError("Native AdamW step must be a U64 scalar")
    moments = {name + suffix: value.shape for name, value in parameters.items() for suffix in (".m", ".v")}
    expected = {} if step.item() == 0 else moments
    if set(state) != {"step", "learning_rate", *expected}:
        raise ValueError("Native AdamW moment inventory differs from its parameter bindings")
    for name, shape in {"learning_rate": (), **expected}.items():
        value = state[name]
        if value.dtype != mx.float32 or value.shape != shape or not np.isfinite(np.asarray(value)).all():
            raise ValueError("Native AdamW tensor representation mismatch")
    if state["learning_rate"].view(mx.uint32).item() != float32.word(settings.learning_rate):
        raise ValueError("Native optimizer learning rate differs from the requested FP32 value")
    random = snapshot["rng"]
    if len(random) != 1 or random[0].dtype != mx.uint32 or random[0].shape != KEY_SHAPE:
        raise ValueError("Expected the native MLX PRNG key")


def signature(snapshot):
    metadata = {name: value for name, value in snapshot.items() if name not in ("optimizer", "rng")}
    metadata["optimizer"] = snapshot["optimizer"]["configuration"]
    result = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    tensors = {"optimizer/" + name: value for name, value in snapshot["optimizer"]["state"].items()}
    tensors.update({"rng/" + str(index): value for index, value in enumerate(snapshot["rng"])})
    for name, value in sorted(tensors.items()):
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(mlx_tensors.view(value))
    return result.hexdigest()


def restore(learner, snapshot, *, policy, identities, settings):
    validate(snapshot, adapter(learner.model), identities=identities, settings=settings)
    if mlx_tensors.digest(policy) != identities["adapter"]:
        raise ValueError("Native restored adapter differs from the requested policy")
    install(learner.model, policy)
    learner.optimizer.state = tree_unflatten(list(snapshot["optimizer"]["state"].items()))
    restore_random(snapshot["rng"])
    mx.eval(learner.model.trainable_parameters(), learner.optimizer.state, mx.random.state)
    actual = mlx_checkpoint.observe(learner.optimizer, identities=identities, parameters=adapter(learner.model))
    if signature(actual) != signature(snapshot):
        raise RuntimeError("Actual restored native optimizer or PRNG differs from its checkpoint")


def restore_random(keys):
    high, low = keys[0].tolist()
    seed = (high << KEY_WORD_BITS) | low
    if not mlx_tensors.equal({"key": mx.random.key(seed)}, {"key": keys[0]}):
        raise ValueError("Native PRNG key differs from the declared two-word seed representation")
    mx.random.seed(seed)
