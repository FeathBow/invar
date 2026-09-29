from dataclasses import asdict
from functools import partial
import json

import mlx.core as mx
import mlx.optimizers as optim

from worker.exchange import Exchange
from worker.logical import Learner, plan
from worker.mlx import adapter as mlx_adapter
from worker.mlx import backward as mlx_backward
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import learning as mlx_learning
from worker.mlx import model as mlx_model
from worker.mlx.training import learning, logprobs
from worker.mlx import state as mlx_state
from worker.mlx import tensors as mlx_tensors
from worker.mlx import tokenization as mlx_tokenization
from worker.record import save as save_probabilities
from worker import registry
from worker.implementation import file_digest
from worker.trajectory import Request, Trajectory
from worker.update import consumed, snapshot
from worker.implementation import LEARNING
from worker.cohort import validate


def optimizer(settings):
    return optim.AdamW(learning_rate=settings.learning_rate, betas=settings.betas, eps=settings.epsilon,
                       weight_decay=settings.weight_decay, bias_correction=True)


def restore(runtime, request, paths):
    validate(runtime.tokenizer, request.samples, encode=mlx_tokenization.prompt)
    policy = mlx_tensors.policy(paths.checkpoint / "adapter.safetensors", request.policy)
    reference = mlx_tensors.policy(paths.reference, request.reference)
    actual, encoded = snapshot(paths.checkpoint / "learner.pt")
    if actual != request.learner:
        raise ValueError("Native learner file differs from its requested checkpoint bytes")
    learner = Learner(model=runtime.model, optimizer=optimizer(request.optimizer), evaluate=logprobs)
    identities = {"adapter": request.policy, **{name: getattr(request, name) for name in ("base", "assembly", "tokenizer")}}
    mlx_state.restore(learner, mlx_checkpoint.load(encoded), policy=policy, identities=identities, settings=request.optimizer)
    return learner, reference


def loaded(request):
    return {name: getattr(request, name) for name in ("policy", "learner", "reference", "tokenizer", "base", "assembly")} | {"optimizer": asdict(request.optimizer)}


def trajectory(item):
    request = Request(sample=item.sample, group=item.group, prompt=item.prompt, seed=item.seed,
                      limit=item.limit, temperature=item.temperature)
    return Trajectory(request=request, tokens=mx.array([item.tokens], dtype=mx.int32),
                      prompt_length=item.prompt_length, behavior=mx.array(item.behavior_bits, dtype=mx.uint32).view(mx.float32),
                      text=item.text, truncated=item.truncated)


def consume(call, *, emit, runtime):
    observed = loaded(call.request)
    emit("loaded_learner", {"binding": call.invocation.binding(), "state": observed, "load": registry.invocation(call.load),
                            "image": registry.learning(observed), "model": runtime.identity[0], "revision": runtime.identity[1]})
    trajectories = tuple(trajectory(item) for item in call.request.samples)
    actual = consumed(call.request, trajectories=trajectories, loaded=observed)
    emit("consumed", {"binding": call.invocation.binding(), "program": call.invocation.program,
                      "request": actual, "load": registry.invocation(call.load)})
    return trajectories, actual


def save(runtime, learner, output, *, expected=None):
    identities = mlx_model.identities(runtime, LEARNING)
    if expected is not None and identities != expected:
        raise RuntimeError("Native successor differs from its observed updated materialization")
    parameters = mlx_adapter.state(learner.model)
    training = mlx_checkpoint.observe(learner.optimizer, identities=identities, parameters=parameters)
    policy = mlx_tensors.save_policy(output / "adapter.safetensors", parameters)
    mlx_checkpoint.save(output / "learner.pt", training)
    return {"policy": policy, "learner": file_digest(output / "learner.pt"), **identities}


def execute(runtime, learner, call, output, *, trajectories, actual, measure, emit, receive):
    request = call.request
    bound = call.invocation.binding()
    planned = plan(request, trajectories, Exchange(binding=bound, emit=emit, receive=receive))
    with learning(runtime.numerics, learner.model):
        result = measure("reward_update", partial(mlx_learning.update, learner, planned, linearize=mlx_backward.linearize))
    if result.summary["before"] != request.policy:
        raise RuntimeError("Native update input differs from the declared policy")
    gradients = output / "gradients.safetensors"

    def artifacts():
        digest = save_probabilities(output / "probabilities.json", request.order, result,
                                    invocation={"binding": bound, "program": call.invocation.program}, request=actual)
        with gradients.open("xb") as target:
            mx.save_safetensors(target, result.gradients, metadata={"binding": json.dumps(bound, sort_keys=True),
                                "program": call.invocation.program, "policy": request.policy,
                                "observation": "objective and reward gradients before AdamW"})
        return digest, file_digest(gradients)

    probability, gradient_digest = measure("artifacts", artifacts)
    expected = {"adapter": result.summary["after"], **{name: getattr(request, name) for name in ("base", "assembly", "tokenizer")}}
    saved = measure("checkpoint", partial(save, runtime, learner, output, expected=expected))
    return {"binding": bound, "request": actual, "update": result.summary,
            "gradients": gradient_digest, "probabilities": probability,
            "adapter": saved["policy"], "learner": saved["learner"], "storage": "staged; not published"}
