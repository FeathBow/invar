from dataclasses import asdict, replace
from pathlib import Path

from worker.hf.artifact import read_checkpoint
from worker.vllm.lora import Target, activate as install, create
from worker.vllm.residency import available, released
from worker.vllm.runtime import targets
from worker.vllm.worker import manager


def install_checkpoint(worker, *, previous, directory, template, receipt, mapping, expected):
    released(worker, adapter_id=previous)
    package = read_checkpoint(Path(directory), template=Path(template), expected=receipt)
    if dict(package.source) != expected:
        raise ValueError("Native product checkpoint differs from the selected policy or frozen learner materialization")
    resolved = tuple(Target(**value) for value in mapping)
    native = manager(worker)
    adapter_id = previous + 1
    model = create(native, package, targets=resolved, adapter_id=adapter_id)
    if not native.remove_adapter(previous):
        raise RuntimeError("Native product did not remove its previous adapter")
    if native.list_adapters() or native._active_adapters or any(value is not None for value in native.lora_index_to_id):
        raise RuntimeError("Native product removal retained an adapter registration or slot")
    if not native.add_adapter(model):
        raise RuntimeError("Native product did not register its published adapter")
    consumed = install(native, package, targets=resolved, adapter_id=adapter_id)
    worker._invar_lora_bindings = {adapter_id: (package, resolved)}
    return {"resident": asdict(consumed), "source": dict(package.source)}


def activate(runtime, directory, *, policy, config, selection_factory):
    available(runtime)
    expected = {**dict(runtime.source), "adapter": policy}
    actual, = runtime.engine.collective_rpc(install_checkpoint, kwargs={
        "previous": runtime.lora.lora_int_id, "directory": str(directory), "template": config.template,
        "receipt": config.handoff, "mapping": [asdict(value) for value in targets(runtime.engine)], "expected": expected})
    consumed = actual["resident"]
    if consumed["policy"] != policy or actual["source"] != expected:
        raise RuntimeError("Native product installed a different published adapter")
    selection = selection_factory(lora_name=consumed["package"], lora_int_id=consumed["adapter_id"], lora_path=str(directory))
    # Only unmerged adapter buffers change; the model stays in the same owned lifetime.
    identities = {**dict(runtime.identities), "adapter": policy}
    return replace(runtime, lora=selection, receipt=consumed["package"],
                   source=tuple(actual["source"].items()), identities=tuple(identities.items()))
