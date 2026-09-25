from dataclasses import asdict
from pathlib import Path

from worker.hf.artifact import read, read_checkpoint
from worker.vllm.lora import Target, activate, create, layout
from worker.vllm.worker import REFERENCE_ID, identified, manager


def registered(worker, *, adapter_id):
    return {adapter_id} | ({REFERENCE_ID} if REFERENCE_ID in getattr(worker, "_invar_lora_bindings", {}) else set())


def idle(worker, *, adapter_id):
    native = manager(worker)
    if getattr(worker, "_invar_execution", None) is not None:
        raise RuntimeError("Native resident policy still owns an execution")
    expected = registered(worker, adapter_id=adapter_id)
    if set(native.list_adapters()) != expected or set(worker._invar_lora_bindings) != expected:
        raise ValueError("Native residence requires exactly its current policy and reference registrations")
    if set(native._active_adapters) != expected:
        raise ValueError("Native residence has a different active policy inventory")
    if sorted(value for value in native.lora_index_to_id if value is not None) != sorted(expected):
        raise ValueError("Native residence has a different occupied slot inventory")


def inspect(worker, *, receipt, adapter_id):
    idle(worker, adapter_id=adapter_id)
    return identified(worker, receipt=receipt, adapter_id=adapter_id)


def released(worker, *, adapter_id):
    idle(worker, adapter_id=adapter_id)
    runner = worker.model_runner
    if runner.requests or runner.input_batch.req_id_to_index or runner.input_batch.num_reqs:
        raise RuntimeError("Native release retained cached request or sampler rows")
    if runner.num_prompt_logprobs or runner.encoder_cache or runner.execute_model_state is not None:
        raise RuntimeError("Native release retained per-request execution state")


def candidate(worker, *, directory, template, receipt, targets, adapter_id, expected):
    from worker.vllm.profile import observe

    package = (read(Path(directory), expected=receipt) if template is None else
               read_checkpoint(Path(directory), template=Path(template), expected=receipt))
    if any(package.source[name] != expected[name] for name in ("adapter", "tokenizer")):
        raise ValueError("Replacement policy differs from the requested materialization")
    mapping = tuple(Target(**value) for value in targets)
    native = manager(worker)
    layout(native, package, mapping)
    model = create(native, package, targets=mapping, adapter_id=adapter_id)
    observed, = observe(worker.model_runner, native, policies={adapter_id: (package, mapping)})
    if (observed.base, observed.assembly) != (expected["base"], expected["assembly"]):
        raise ValueError("Replacement native model differs from the requested materialization")
    return package, mapping, model


def replace(worker, *, previous, directory, template, receipt, targets, adapter_id, expected):
    idle(worker, adapter_id=previous)
    if type(adapter_id) is not int or adapter_id <= previous:
        raise ValueError("Replacement requires a fresh native adapter ID")
    package, mapping, model = candidate(worker, directory=directory, template=template, receipt=receipt,
                                        targets=targets, adapter_id=adapter_id, expected=expected)
    native = manager(worker)
    kept = registered(worker, adapter_id=previous) - {previous}
    if not native.remove_adapter(previous):
        raise RuntimeError("Native policy removal did not release the previous registration")
    remaining = {value for value in native.lora_index_to_id if value is not None}
    if set(native.list_adapters()) != kept or set(native._active_adapters) != kept or remaining != kept:
        raise RuntimeError("Native policy removal retained a registration or occupied slot")
    if not native.add_adapter(model):
        raise RuntimeError("Native replacement did not register the checked policy")
    activate(native, package, targets=mapping, adapter_id=adapter_id)
    worker._invar_lora_bindings = {**{key: value for key, value in worker._invar_lora_bindings.items() if key in kept},
                                   adapter_id: (package, mapping)}
    return inspect(worker, receipt=package.identity, adapter_id=adapter_id)


def available(runtime):
    engine = runtime.engine.llm_engine
    if engine.has_unfinished_requests() or engine.output_processor.request_states:
        raise RuntimeError("Native resident runtime still owns queued requests")


def scheduler(runtime):
    from vllm.v1.engine.core_client import InprocClient

    client = runtime.engine.llm_engine.engine_core
    if not isinstance(client, InprocClient):
        raise ValueError("Native residence requires the directly owned synchronous engine client")
    return client.engine_core.scheduler


def release(runtime):
    available(runtime)
    owned = scheduler(runtime)
    if owned.has_unfinished_requests() or owned.requests:
        raise RuntimeError("Native release still has unfinished scheduler requests")
    # The final output precedes native removal of finished worker request rows.
    if runtime.engine.llm_engine.step():
        raise RuntimeError("Native release unexpectedly produced request outputs")
    if owned.has_requests():
        raise RuntimeError("Native release retained scheduler work")
    if any(group.req_to_blocks for group in owned.kv_cache_manager.coordinator.single_type_managers):
        raise RuntimeError("Native release retained request-owned attention or recurrent cache blocks")
    runtime.engine.collective_rpc(released, kwargs={"adapter_id": runtime.lora.lora_int_id})


def select(runtime, directory, *, expected, config, selection_factory):
    from worker.hf.operation import verify
    from worker.vllm.runtime import bound, selection, targets

    available(runtime)
    verify(runtime.tokenizer, dict(runtime.identities)["tokenizer"])
    adapter_id = runtime.lora.lora_int_id
    current, = runtime.engine.collective_rpc(inspect, kwargs={"receipt": runtime.receipt, "adapter_id": adapter_id})
    actual = {"adapter": current["resident"]["policy"], "tokenizer": current["source"]["tokenizer"],
              "base": current["model"]["base"], "assembly": current["model"]["assembly"]}
    bound(actual, dict(runtime.identities))
    if actual["adapter"] == expected["adapter"]:
        bound(actual, expected)
        return runtime
    verify(runtime.tokenizer, expected["tokenizer"])
    observed, = runtime.engine.collective_rpc(replace, kwargs={
        "previous": adapter_id, "adapter_id": adapter_id + 1, "directory": str(directory),
        "template": config.template, "receipt": config.handoff, "expected": expected,
        "targets": [asdict(value) for value in targets(runtime.engine)]})
    selected = selection(runtime.engine, runtime.tokenizer, observed=observed, directory=directory,
                         config=config, selection_factory=selection_factory)
    bound(dict(selected.identities), expected)
    return selected
