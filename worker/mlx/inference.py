from functools import partial

import mlx.core as mx

from worker.batch import FORMAT, capture
from worker.report import ready, result
from worker.mlx import adapter as mlx_adapter
from worker.mlx.crossscore import score
from worker.mlx import model as mlx_model
from worker.mlx.rollout import generate
from worker.mlx import tensors as mlx_tensors
from worker.scoring import TokenPath
from worker.implementation import INFERENCE


def execute(runtime, call, *, approve, measure, emit, sampling, previous=None):
    identities = mlx_model.verify(runtime, call.identities, INFERENCE)
    ready(call, identities, model=runtime.identity, previous=previous, emit=emit)
    approve(call.invocation)
    trajectory, = measure("inference", lambda: sample(runtime, (call.request,), sampling=sampling))
    result(call, trajectory, identities=identities, emit=emit)


def execute_batch(runtime, calls, *, approve, measure, emit, sampling, reference=None):
    expected = calls[0].identities
    if any(call.identities != expected for call in calls):
        raise ValueError("Native batch members require the same model materialization")
    identities = mlx_model.verify(runtime, expected, INFERENCE)
    readiness = [capture(partial(ready, call, identities, model=runtime.identity, previous=None)) for call in calls]
    emit("consumed", {"format": FORMAT, "calls": readiness})
    approve(tuple(call.invocation for call in calls))
    requests = tuple(call.request for call in calls)
    trajectories, scores = measure("inference", lambda: scored(runtime, sample(runtime, requests, sampling=sampling),
                                                               reference=reference, sampling=sampling))
    completed = [capture(partial(result, call, trajectory, identities=identities, reference=score))
                 for call, trajectory, score in zip(calls, trajectories, scores, strict=True)]
    emit("result", {"format": FORMAT, "calls": completed})


def scored(runtime, trajectories, *, reference, sampling):
    if reference is None:
        return trajectories, (None,) * len(trajectories)
    current = mlx_adapter.state(runtime.model)
    mlx_adapter.install(runtime.model, mlx_tensors.policy(reference.adapter, reference.digest))
    try:
        paths = tuple(TokenPath(prefix=tuple(item.tokens[0, :item.prompt_length].tolist()),
                                response=tuple(item.tokens[0, item.prompt_length:].tolist())) for item in trajectories)
        observed = score(runtime.model, runtime.tokenizer, tuple(item.request for item in trajectories),
                         paths=paths, sampling=sampling)
    finally:
        mlx_adapter.install(runtime.model, current)
    if not mlx_tensors.equal(current, mlx_adapter.state(runtime.model)):
        raise RuntimeError("Native reference scoring did not restore the sampled policy")
    return trajectories, tuple((reference.digest, item.log_probability_bits) for item in observed)


def sample(runtime, requests, *, sampling):
    before = {str(index): mx.array(value) for index, value in enumerate(mx.random.state)}
    values = generate(runtime.model, runtime.tokenizer, requests, sampling=sampling)
    after = {str(index): value for index, value in enumerate(mx.random.state)}
    if not mlx_tensors.equal(before, after):
        raise RuntimeError("Native inference advanced the learner PRNG state")
    return values
