from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from functools import partial

from worker.batch import FORMAT, capture
from worker.report import ready, result
from worker.hf.operation import verify
from worker.vllm.configuration import snapshot
from worker.vllm.execution import generate
from worker.vllm.lora import qwen_mlp_targets
from worker.vllm.worker import REFERENCE_ID, identified, prepare, prepare_checkpoint

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


def execute(runtime, call, *, approve, measure, emit, previous=None, reference=None, config=None, selection_factory=None):
    tokenizer = verify(runtime.tokenizer, call.identities["tokenizer"])
    scoring = None if reference is None else referenced(runtime, reference, config=config, selection_factory=selection_factory)

    def permission(prepared):
        identities = observed(prepared, runtime, tokenizer=tokenizer)
        bound(identities, call.identities)
        verify(runtime.tokenizer, tokenizer)
        ready(call, identities, model=runtime.identity, previous=previous, emit=emit)
        approve(call.invocation)
        verify(runtime.tokenizer, tokenizer)

    def sampled():
        execution = generate(runtime.engine, runtime.tokenizer, [call.request],
                             loras=[runtime.lora], receipts=[runtime.receipt], approve=permission)
        if scoring is None:
            return execution, None
        expected = {**dict(runtime.identities), "adapter": reference.digest}
        score, = scored(scoring, execution.trajectories, tokenizer=tokenizer, expected=expected)
        return execution, score

    execution, score = measure("inference", sampled)
    verify(runtime.tokenizer, tokenizer)
    trajectory, = execution.trajectories
    identities = observed(execution.prepared, runtime, tokenizer=tokenizer)
    bound(identities, call.identities)
    result(call, trajectory, identities=identities, emit=emit, reference=score)


def referenced(runtime, reference, *, config, selection_factory):
    arguments = {"receipt": config.handoff, "targets": [asdict(value) for value in targets(runtime.engine)],
                 "adapter_id": REFERENCE_ID}
    if config.template is None:
        installed, = runtime.engine.collective_rpc(prepare, kwargs={**arguments, "directory": str(reference.adapter)})
    else:
        installed, = runtime.engine.collective_rpc(prepare_checkpoint, kwargs={**arguments, "adapter": str(reference.adapter),
                                                                               "template": config.template})
    if installed["policy"] != reference.digest:
        raise ValueError("Resident reference adapter differs from its declared identity")
    lora = selection_factory(lora_name=installed["package"], lora_int_id=REFERENCE_ID, lora_path=str(reference.adapter))
    return replace(runtime, lora=lora, receipt=installed["package"])


def scored(scoring, trajectories, *, tokenizer, expected):
    from worker.scoring import TokenPath
    from worker.vllm.crossscore import score

    paths = [TokenPath(prefix=tuple(item.tokens[0, :item.prompt_length].tolist()),
                       response=tuple(item.tokens[0, item.prompt_length:].tolist())) for item in trajectories]

    def check(prepared):
        if observed(prepared, scoring, tokenizer=tokenizer) != expected:
            raise ValueError("Native reference scoring prepared a different adapter or model")

    count = len(paths)
    execution = score(scoring.engine, scoring.tokenizer, [item.request for item in trajectories], paths=paths,
                      loras=[scoring.lora] * count, receipts=[scoring.receipt] * count, approve=check)
    return tuple((expected["adapter"], item.log_probability_bits) for item in execution.scores)


def execute_batch(runtime, calls, *, approve, measure, emit, reference=None, config=None, selection_factory=None):
    calls = tuple(calls)
    if not calls:
        raise ValueError("Native batching requires at least one independently bound call")
    for call in calls:
        bound(dict(runtime.identities), call.identities)
    tokenizer = verify(runtime.tokenizer, calls[0].identities["tokenizer"])
    scoring = None if reference is None else referenced(runtime, reference, config=config, selection_factory=selection_factory)

    def permission(prepared):
        identities = observed(prepared, runtime, tokenizer=tokenizer)
        for call in calls:
            bound(identities, call.identities)
        verify(runtime.tokenizer, tokenizer)
        readiness = [capture(partial(ready, call, identities, model=runtime.identity, previous=None)) for call in calls]
        emit("consumed", {"format": FORMAT, "calls": readiness})
        approve(tuple(call.invocation for call in calls))
        verify(runtime.tokenizer, tokenizer)

    def sampled():
        execution = generate(runtime.engine, runtime.tokenizer, [call.request for call in calls],
                             loras=[runtime.lora] * len(calls), receipts=[runtime.receipt] * len(calls), approve=permission)
        if scoring is None:
            return execution, (None,) * len(calls)
        expected = {**dict(runtime.identities), "adapter": reference.digest}
        return execution, scored(scoring, execution.trajectories, tokenizer=tokenizer, expected=expected)

    execution, scores = measure("inference", sampled)
    verify(runtime.tokenizer, tokenizer)
    identities = observed(execution.prepared, runtime, tokenizer=tokenizer)
    for call in calls:
        bound(identities, call.identities)
    completed = [capture(partial(result, call, trajectory, identities=identities, reference=score))
                 for call, trajectory, score in zip(calls, execution.trajectories, scores, strict=True)]
    emit("result", {"format": FORMAT, "calls": completed})
    return execution
