from dataclasses import asdict
from functools import partial
from pathlib import Path

from worker.hf.artifact import read, read_checkpoint
from worker.vllm.lora import Target, activate, verify
from worker.vllm.rollout import LOGPROBS_MODE


def manager(worker):
    runner = worker.model_runner
    native = runner.lora_manager._adapter_manager
    if runner.get_model() is not native.model:
        raise ValueError("Native runner and LoRA manager do not own the same model")
    modes = (worker.model_config.logprobs_mode, runner.sampler.logprobs_mode,
             runner.sampler.topk_topp_sampler.logprobs_mode)
    if any(mode != LOGPROBS_MODE for mode in modes):
        raise ValueError("Native worker must sample with processed behavior log probabilities")
    return native


def prepare(worker, *, directory, receipt, targets, adapter_id):
    package = read(Path(directory), expected=receipt)
    return prepare_package(worker, package, targets=targets, adapter_id=adapter_id)


def prepare_checkpoint(worker, *, adapter, template, receipt, targets, adapter_id):
    package = read_checkpoint(Path(adapter), template=Path(template), expected=receipt)
    return prepare_package(worker, package, targets=targets, adapter_id=adapter_id)


def prepare_package(worker, package, *, targets, adapter_id):
    native = manager(worker)
    mapping = tuple(Target(**target) for target in targets)
    bindings = getattr(worker, "_invar_lora_bindings", {})
    if adapter_id in bindings and bindings[adapter_id][0].identity != package.identity:
        raise ValueError("Native adapter ID already belongs to another handoff")
    consumed = activate(native, package, targets=mapping, adapter_id=adapter_id)
    worker._invar_lora_bindings = {**bindings, adapter_id: (package, mapping)}
    return asdict(consumed)


def consumed(worker, *, receipt, adapter_id):
    native = manager(worker)
    bindings = getattr(worker, "_invar_lora_bindings", {})
    if adapter_id not in bindings or bindings[adapter_id][0].identity != receipt:
        raise ValueError("Native worker has not prepared the requested handoff")
    package, mapping = bindings[adapter_id]
    return asdict(verify(native, package, targets=mapping, adapter_id=adapter_id))


def identified(worker, *, receipt, adapter_id):
    from worker.vllm.profile import inspection

    resident = consumed(worker, receipt=receipt, adapter_id=adapter_id)
    policy = worker._invar_lora_bindings[adapter_id]
    models, profile = inspection(worker.model_runner, manager(worker), policies={adapter_id: policy})
    model, = models
    worker._invar_observed = (frozenset({adapter_id}), models)
    return {"resident": resident, "model": asdict(model), "source": dict(policy[0].source), "profile": profile}


def begin(worker, *, bindings):
    from worker.vllm.state import Monitor, binding
    from worker.vllm.profile import observe

    if getattr(worker, "_invar_execution", None) is not None:
        raise RuntimeError("Native worker already owns an execution")
    native = manager(worker)
    selected = {value["selection"]["adapter"] for value in bindings}
    policies = {key: value for key, value in worker._invar_lora_bindings.items() if key in selected}
    observed = getattr(worker, "_invar_observed", None)
    monitor = Monitor(worker.model_runner, native, bindings=tuple(binding(value) for value in bindings),
                      verify=partial(consumed, worker),
                      observe_model=partial(observe, worker.model_runner, native, policies=policies),
                      models=observed[1] if observed is not None and observed[0] == frozenset(policies) else None)
    monitor.attach()
    worker._invar_execution = monitor
    return {"resident": monitor.resident, "models": tuple(asdict(value) for value in monitor.models)}


def permit(worker):
    monitor = worker._invar_execution
    if monitor is None or monitor.permitted:
        raise RuntimeError("Native execution permission is absent or already consumed")
    monitor.allow()


def end(worker, *, completed):
    monitor = worker._invar_execution
    if monitor is None:
        raise RuntimeError("Native worker has no owned execution to finish")
    worker._invar_execution = None
    return monitor.close(completed=completed)
