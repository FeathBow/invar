from dataclasses import dataclass, replace
import gc

import mlx.core as mx

from worker.advantage import check
from worker.mlx import adapter as mlx_adapter
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import model as mlx_model
from worker.mlx import state as mlx_state
from worker.mlx import step as mlx_step
from worker.mlx import tensors as mlx_tensors
from worker.mlx import tokenization as mlx_tokenization
from worker.update import snapshot


@dataclass(frozen=True, kw_only=True)
class Witness:
    policy: str
    learner: str
    signature: str


@dataclass(frozen=True, kw_only=True)
class Runtime:
    loaded: mlx_model.Loaded
    learner: object
    reference: dict
    reference_identity: str
    settings: object
    saved: Witness


def observe(loaded, learner):
    identities = mlx_model.identities(loaded)
    return mlx_checkpoint.observe(learner.optimizer, identities=identities, parameters=mlx_adapter.state(learner.model))


def attest(loaded, learner, directory, *, policy, expected):
    identity, encoded = snapshot(directory / "learner.pt")
    if identity != expected:
        raise ValueError("Native checkpoint differs from its declared learner file identity")
    saved = mlx_checkpoint.load(encoded)
    actual = observe(loaded, learner)
    if saved["adapter"] != policy or actual["adapter"] != policy or mlx_state.signature(saved) != mlx_state.signature(actual):
        raise RuntimeError("Actual native learner differs from its saved checkpoint")
    return Witness(policy=policy, learner=identity, signature=mlx_state.signature(saved))


def restore(loaded, request, paths):
    learner, reference = mlx_step.restore(loaded, request, paths)
    saved = attest(loaded, learner, paths.checkpoint, policy=request.policy, expected=request.learner)
    return Runtime(loaded=loaded, learner=learner, reference=reference,
                   reference_identity=request.reference, settings=request.optimizer, saved=saved)


def verify(runtime):
    if any(module.training for _, module in runtime.loaded.model.named_modules()):
        raise RuntimeError("Native resident learner retains training mode outside an update")
    actual = observe(runtime.loaded, runtime.learner)
    mlx_state.validate(actual, mlx_adapter.state(runtime.learner.model),
                       identities={"adapter": runtime.saved.policy}, settings=runtime.settings)
    if mlx_state.signature(actual) != runtime.saved.signature:
        raise RuntimeError("Live native optimizer or PRNG differs from the saved successor")
    if mlx_tensors.digest(runtime.reference) != runtime.reference_identity:
        raise RuntimeError("Native resident reference differs from its fixed original policy")
    return {name: actual[name] for name in ("adapter", "tokenizer", "base", "assembly")}


def activate(runtime, request, paths):
    if (request.policy, request.learner, request.reference, request.optimizer) != (runtime.saved.policy, runtime.saved.learner, runtime.reference_identity, runtime.settings):
        raise ValueError("Native resident update must consume its completed successor with the original optimizer")
    # The publication path must still name the actual completed bytes; retain
    # live optimizer and PRNG values rather than restoring them on activation.
    if snapshot(paths.checkpoint / "learner.pt")[0] != request.learner:
        raise ValueError("Native successor path differs from the published learner bytes")
    mlx_tensors.policy(paths.checkpoint / "adapter.safetensors", request.policy)
    actual = verify(runtime)
    expected = {"adapter": request.policy, **{name: getattr(request, name) for name in ("base", "assembly", "tokenizer")}}
    if actual != expected:
        raise ValueError("Native resident materialization differs from its requested update")
    mlx_tokenization.validate(runtime.loaded.tokenizer, request.samples)
    return runtime


def execute(runtime, call, output, *, measure, approve, emit):
    checked = check(call.request)
    admitted, actual = mlx_step.consume(runtime.learner, call, runtime.reference,
                                        checked=checked, measure=measure, emit=emit, runtime=runtime.loaded)
    approve(call.invocation)
    verify(runtime)
    result = mlx_step.execute(runtime.loaded, runtime.learner, call, output, admitted=admitted,
                              actual=actual, measure=measure)
    saved = attest(runtime.loaded, runtime.learner, output, policy=result["adapter"], expected=result["learner"])
    emit("result", result)
    return replace(runtime, saved=saved)


def release(runtime):
    gc.collect()
    mx.clear_cache()
    verify(runtime)
