from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import partial

from worker.batch import FORMAT, capture
from worker.report import ready, result
from worker.hf.operation import verify
from worker.vllm.configuration import snapshot
from worker.vllm.execution import generate
from worker.vllm.lora import qwen_mlp_targets
from worker.vllm.worker import identified, prepare, prepare_checkpoint

ADAPTER_ID = 1
SOURCE_PREFIX = "model.language_model.layers"
RUNTIME_PREFIX = "language_model.model.layers"


@dataclass(frozen=True, kw_only=True)
class Runtime:
    engine: object
    tokenizer: object
    lora: object
    receipt: str
    identity: tuple[str, str]
    source: tuple[tuple[str, str], ...]
    identities: tuple[tuple[str, str], ...]


def targets(engine):
    layers = engine.llm_engine.model_config.hf_text_config.num_hidden_layers
    return qwen_mlp_targets(layers=layers, source_prefix=SOURCE_PREFIX, runtime_prefix=RUNTIME_PREFIX)


def materialize(engine, directory, *, config):
    arguments = {"receipt": config.handoff, "targets": [asdict(value) for value in targets(engine)], "adapter_id": ADAPTER_ID}
    if config.template is None:
        installed, = engine.collective_rpc(prepare, kwargs={**arguments, "directory": str(directory)})
    else:
        installed, = engine.collective_rpc(prepare_checkpoint, kwargs={**arguments, "adapter": str(directory), "template": config.template})
    observed, = engine.collective_rpc(identified, kwargs={"receipt": installed["package"], "adapter_id": ADAPTER_ID})
    return observed


def selection(engine, tokenizer, *, observed, directory, config, selection_factory):
    consumed = observed["resident"]
    model = observed["model"]
    adapter_id = consumed["adapter_id"]
    if model["adapter_id"] != adapter_id:
        raise ValueError("Native materialization identified another adapter")
    tokenizer_id = verify(tokenizer, observed["source"]["tokenizer"])
    identities = {"adapter": consumed["policy"], "tokenizer": tokenizer_id,
                  "base": model["base"], "assembly": model["assembly"]}
    return Runtime(engine=engine, tokenizer=tokenizer,
                   lora=selection_factory(lora_name=consumed["package"], lora_int_id=adapter_id, lora_path=str(directory)),
                   receipt=consumed["package"], identity=(config.model, config.revision),
                   source=tuple(observed["source"].items()), identities=tuple(identities.items()))


def configured(engine, tokenizer, *, directory, config, selection_factory, emit):
    observed = materialize(engine, directory, config=config)
    runtime = selection(engine, tokenizer, observed=observed, directory=directory, config=config,
                        selection_factory=selection_factory)
    emit("profile", {"model": config.model, "revision": config.revision, "native": observed["profile"]})
    return runtime


@contextmanager
def load(cache, adapter, *, config, factory, tokenizer_loader, resolve, selection_factory, measure, emit):
    path = snapshot(config, cache, resolve=resolve)
    tokenizer = tokenizer_loader(path)
    with ExitStack() as stack:
        def prepare():
            emit("loading", {"model": config.model, "revision": config.revision})
            engine = factory(model=path, **config.arguments())
            stack.callback(engine.llm_engine.engine_core.shutdown)
            return configured(engine, tokenizer, directory=adapter, config=config,
                              selection_factory=selection_factory, emit=emit)

        yield measure("load", prepare)


def observed(prepared, runtime, *, tokenizer):
    resident, = prepared.resident
    model, = prepared.models
    policy, = resident
    loaded, = model
    adapter_id = runtime.lora.lora_int_id
    if (policy.package, policy.adapter_id, loaded.adapter_id) != (runtime.receipt, adapter_id, adapter_id):
        raise ValueError("Native execution prepared a different adapter selection")
    return {"adapter": policy.policy, "tokenizer": tokenizer, "base": loaded.base, "assembly": loaded.assembly}


def bound(identities, expected):
    if identities != expected:
        raise ValueError("Native loaded identities differ from the requested materialization")


def execute(runtime, call, *, approve, measure, emit, previous=None):
    tokenizer = verify(runtime.tokenizer, call.identities["tokenizer"])

    def permission(prepared):
        identities = observed(prepared, runtime, tokenizer=tokenizer)
        bound(identities, call.identities)
        verify(runtime.tokenizer, tokenizer)
        ready(call, identities, model=runtime.identity, previous=previous, emit=emit)
        approve(call.invocation)
        verify(runtime.tokenizer, tokenizer)

    execution = measure("inference", lambda: generate(runtime.engine, runtime.tokenizer, [call.request],
                                                       loras=[runtime.lora], receipts=[runtime.receipt], approve=permission))
    verify(runtime.tokenizer, tokenizer)
    trajectory, = execution.trajectories
    identities = observed(execution.prepared, runtime, tokenizer=tokenizer)
    bound(identities, call.identities)
    result(call, trajectory, identities=identities, emit=emit)


def execute_batch(runtime, calls, *, approve, measure, emit):
    calls = tuple(calls)
    if not calls:
        raise ValueError("Native batching requires at least one independently bound call")
    for call in calls:
        bound(dict(runtime.identities), call.identities)
    tokenizer = verify(runtime.tokenizer, calls[0].identities["tokenizer"])

    def permission(prepared):
        identities = observed(prepared, runtime, tokenizer=tokenizer)
        for call in calls:
            bound(identities, call.identities)
        verify(runtime.tokenizer, tokenizer)
        readiness = [capture(partial(ready, call, identities, model=runtime.identity, previous=None)) for call in calls]
        emit("consumed", {"format": FORMAT, "calls": readiness})
        approve(tuple(call.invocation for call in calls))
        verify(runtime.tokenizer, tokenizer)

    execution = measure("inference", lambda: generate(runtime.engine, runtime.tokenizer,
                                                       [call.request for call in calls],
                                                       loras=[runtime.lora] * len(calls),
                                                       receipts=[runtime.receipt] * len(calls), approve=permission))
    verify(runtime.tokenizer, tokenizer)
    identities = observed(execution.prepared, runtime, tokenizer=tokenizer)
    for call in calls:
        bound(identities, call.identities)
    completed = [capture(partial(result, call, trajectory, identities=identities))
                 for call, trajectory in zip(calls, execution.trajectories, strict=True)]
    emit("result", {"format": FORMAT, "calls": completed})
    return execution
