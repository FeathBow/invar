from functools import partial

import mlx.core as mx

from worker.batch import FORMAT, capture
from worker.report import ready, result
from worker.mlx import model as mlx_model
from worker.mlx.rollout import generate
from worker.mlx import tensors as mlx_tensors


def execute(runtime, call, *, approve, measure, emit, sampling, previous=None):
    identities = mlx_model.verify(runtime, call.identities)
    ready(call, identities, model=runtime.identity, previous=previous, emit=emit)
    approve(call.invocation)
    trajectory, = measure("inference", lambda: sample(runtime, (call.request,), sampling=sampling))
    result(call, trajectory, identities=identities, emit=emit)


def execute_batch(runtime, calls, *, approve, measure, emit, sampling):
    expected = calls[0].identities
    if any(call.identities != expected for call in calls):
        raise ValueError("Native batch members require the same model materialization")
    identities = mlx_model.verify(runtime, expected)
    readiness = [capture(partial(ready, call, identities, model=runtime.identity, previous=None)) for call in calls]
    emit("consumed", {"format": FORMAT, "calls": readiness})
    approve(tuple(call.invocation for call in calls))
    trajectories = measure("inference", lambda: sample(runtime, tuple(call.request for call in calls), sampling=sampling))
    completed = [capture(partial(result, call, trajectory, identities=identities))
                 for call, trajectory in zip(calls, trajectories, strict=True)]
    emit("result", {"format": FORMAT, "calls": completed})


def sample(runtime, requests, *, sampling):
    before = {str(index): mx.array(value) for index, value in enumerate(mx.random.state)}
    values = generate(runtime.model, runtime.tokenizer, requests, sampling=sampling)
    after = {str(index): value for index, value in enumerate(mx.random.state)}
    if not mlx_tensors.equal(before, after):
        raise RuntimeError("Native inference advanced the learner PRNG state")
    return values
