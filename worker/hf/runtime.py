from dataclasses import dataclass, replace
from functools import partial
import gc

import torch

from worker.exchange import Exchange
from worker.hf import state as learner_state
from worker.hf.learning import Learner, check_optimizer
from worker.hf.step import consume, execute as update, loaded_inputs, optimizer_options, plan, restore_inputs
from worker.hf.tensors import digest
from worker.cohort import validate
from worker.tokenization import prompt


@dataclass(frozen=True, kw_only=True)
class Runtime:
    learner: Learner
    tokenizer: object
    reference: dict[str, torch.Tensor]
    reference_identity: str
    identity: tuple[str, str]
    saved: learner_state.Witness


def initialize(options, request, *, loader, measure, evaluate):
    model, tokenizer, identity = loader(options, request)
    return measure("activation", partial(restore, (model, tokenizer, identity), request,
                                          options=options, evaluate=evaluate))


def restore(loaded, request, *, options, evaluate):
    model, tokenizer, identity = loaded
    optimizer, reference, observed = restore_inputs(model, request, options, tokenizer=tokenizer)
    learner = Learner(model=model, optimizer=optimizer, evaluate=evaluate)
    saved = learner_state.attest(learner, tokenizer, checkpoint=options.checkpoint,
                                 policy=request.policy, expected=request.learner)
    runtime = Runtime(learner=learner, tokenizer=tokenizer, reference={name: value.clone() for name, value in reference.items()},
                      reference_identity=observed["reference"], identity=identity, saved=saved)
    idle(runtime)
    return runtime, observed


def idle(runtime):
    model = runtime.learner.model
    if any(module.training for module in model.modules()) or any(value.grad is not None for value in model.parameters()):
        raise RuntimeError("Resident learner retains an active training mode or gradient")


def verify(runtime, request):
    if (request.policy, request.learner) != (runtime.saved.policy, runtime.saved.learner):
        raise ValueError("Resident update must consume its completed successor checkpoint")
    idle(runtime)
    actual = learner_state.verify(runtime.learner, runtime.tokenizer, runtime.saved)
    reference = digest(runtime.reference)
    if reference != runtime.reference_identity or reference != request.reference:
        raise RuntimeError("Resident reference differs from the fixed original policy")
    if any(actual[name] != getattr(request, name) for name in ("tokenizer", "base", "assembly")):
        raise ValueError("Resident learner materialization differs from the requested update")
    optimizer = runtime.learner.optimizer
    expected = optimizer_options(request.optimizer)
    if any(group[name] != value for group in optimizer.param_groups for name, value in expected.items()):
        raise ValueError("Resident optimizer differs from the declared update specification")
    check_optimizer(optimizer)
    return loaded_inputs(optimizer, {"policy": actual["adapter"], "learner": runtime.saved.learner,
                                     "reference": reference, **{name: actual[name] for name in ("tokenizer", "base", "assembly")}})


def activate(runtime, request):
    observed = verify(runtime, request)
    validate(runtime.tokenizer, request.samples, encode=prompt)
    return observed


def execute(runtime, call, output, *, loaded, measure, permission, emit, receive):
    learner = runtime.learner
    trajectories, actual = consume(call, loaded=loaded, identity=runtime.identity, emit=emit)
    permission(call.invocation)
    verify(runtime, call.request)
    exchange = Exchange(binding=call.invocation.binding(), emit=emit, receive=receive)
    result = update(learner, call, output, plan=plan(call.request, trajectories, exchange), actual=actual,
                    tokenizer=runtime.tokenizer, measure=measure)
    saved = learner_state.attest(learner, runtime.tokenizer, checkpoint=output,
                                 policy=result["adapter"], expected=result["learner"])
    emit("result", result)
    return replace(runtime, saved=saved)


def release(runtime):
    runtime.learner.optimizer.zero_grad(set_to_none=True)
    gc.collect()
    idle(runtime)
    learner_state.verify(runtime.learner, runtime.tokenizer, runtime.saved)
    if digest(runtime.reference) != runtime.reference_identity:
        raise RuntimeError("Resident reference differs from the fixed original policy")


def close():
    gc.collect()
    torch.cuda.empty_cache()
