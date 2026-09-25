from dataclasses import asdict
from functools import partial
import json

import mlx.core as mx
import mlx.optimizers as optim

from worker.logical import Batch, Learner, Sample
from worker.mlx import adapter as mlx_adapter
from worker.mlx import backward as mlx_backward
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import learning as mlx_learning
from worker.mlx import model as mlx_model
from worker.mlx.rollout import logprobs
from worker.mlx import state as mlx_state
from worker.mlx import tensors as mlx_tensors
from worker.mlx import tokenization as mlx_tokenization
from worker.record import save as save_probabilities
from worker import registry
from worker.scalar import Profile
from worker.hf.step import file_digest
from worker.trajectory import Request, Trajectory
from worker.update import consumed, snapshot


def optimizer(settings):
    return optim.AdamW(learning_rate=settings.learning_rate, betas=settings.betas, eps=settings.epsilon,
                       weight_decay=settings.weight_decay, bias_correction=True)


def restore(runtime, request, paths):
    mlx_tokenization.validate(runtime.tokenizer, request.samples)
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


def batch(request, *, checked, measure, emit):
    trajectories = tuple(trajectory(item) for item in request.samples)
    scores = tuple(item.reference_bits for item in request.samples)
    proximal, fixed = measure("probability_roles", partial(mlx_learning.probabilities, trajectories, scores))
    normalized = dict(checked.values)
    samples = tuple(Sample(trajectory=item, proximal=old, reference=ref, advantage=normalized[item.request.sample])
                    for item, old, ref in zip(trajectories, proximal, fixed, strict=True))
    for item in samples:
        emit("roles", {"sample": item.trajectory.request.sample, "proximal_policy": request.policy,
                       "reference_policy": request.reference, "proximal": item.proximal.tolist(),
                       "reference": item.reference.tolist(), "advantage": item.advantage})
    return Batch(samples=samples, order=request.order, profile=Profile(epsilon=request.epsilon, penalty=request.penalty))


def consume(call, *, checked, measure, emit, runtime):
    observed = loaded(call.request)
    emit("loaded_learner", {"binding": call.invocation.binding(), "state": observed, "load": registry.invocation(call.load),
                            "image": registry.learning(observed), "model": runtime.identity[0], "revision": runtime.identity[1]})
    admitted = batch(call.request, checked=checked, measure=measure, emit=emit)
    actual = consumed(call.request, batch=admitted, rewards=checked.rewards, loaded=observed)
    emit("consumed", {"binding": call.invocation.binding(), "program": call.invocation.program,
                      "request": actual, "load": registry.invocation(call.load)})
    return admitted, actual


def save(runtime, learner, output, *, expected=None):
    identities = mlx_model.identities(runtime)
    if expected is not None and identities != expected:
        raise RuntimeError("Native successor differs from its observed updated materialization")
    parameters = mlx_adapter.state(learner.model)
    training = mlx_checkpoint.observe(learner.optimizer, identities=identities, parameters=parameters)
    policy = mlx_tensors.save_policy(output / "adapter.safetensors", parameters)
    mlx_checkpoint.save(output / "learner.pt", training)
    return {"policy": policy, "learner": file_digest(output / "learner.pt"), **identities}


def execute(runtime, learner, call, output, *, admitted, actual, measure):
    request = call.request
    with runtime.numerics.learning(learner.model):
        result = measure("reward_update", partial(mlx_learning.update, learner, admitted, linearize=mlx_backward.linearize))
    if result.summary["before"] != request.policy:
        raise RuntimeError("Native update input differs from the declared policy")
    bound = call.invocation.binding()
    gradients = output / "gradients.safetensors"

    def artifacts():
        digest = save_probabilities(output / "probabilities.json", result.probabilities,
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
