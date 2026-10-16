from dataclasses import dataclass

import torch

from worker.report import ready, result
from worker.hf.policy import activate, read_adapter, verify as verify_adapter
from worker.hf.model import CPU_THREADS, DEFAULT_SEED, load_model, snapshot
from worker.hf.metrics import measure, report
from worker.hf.score import referenced
from worker.hf.rollout import generate
from worker.implementation import INFERENCE
from worker.hf.operation import load as load_tokenizer
from worker.hf.operation import described, verify

@dataclass(frozen=True, kw_only=True)
class Runtime:
    model: torch.nn.Module
    tokenizer: object
    adapter: object
    device: str
    identity: tuple[str, str]


def load(cache, adapter, *, expected, declared, emit=report, role=INFERENCE):
    state = read_adapter(adapter, expected["adapter"])
    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    path = snapshot(cache, declared)
    tokenizer = load_tokenizer(path)
    verify(tokenizer, expected["tokenizer"])
    model = measure("load", lambda: load_model(path, role=role, emit=emit, adapted=bool(state), identity=declared), emit=emit)
    installed(model, state, expected, role=role)
    return Runtime(model=model, tokenizer=tokenizer, adapter=adapter, device="cuda", identity=declared)


def installed(model, state, expected, *, role=INFERENCE):
    return activate(model, state, base=expected["base"], assembly=expected["assembly"], role=role)


def materialized(runtime, requested):
    tokenizer = verify(runtime.tokenizer, requested["tokenizer"])
    consumed = verify_adapter(runtime.model, requested["adapter"],
                              base=requested["base"], assembly=requested["assembly"], role=INFERENCE)
    return {"adapter": consumed, "tokenizer": tokenizer, "base": requested["base"], "assembly": requested["assembly"]}


def sampled(runtime, requests, reference, *, identities):
    trajectories = tuple(generate(runtime.model, runtime.tokenizer, request, device=runtime.device) for request in requests)
    return trajectories, referenced(runtime, trajectories, reference, identities=identities)


def execute(runtime, call, *, approve, measure, previous=None, emit=report, reference=None):
    identities = materialized(runtime, call.identities)
    ready(call, identities, model=runtime.identity, description=described(runtime.tokenizer, identities["tokenizer"]),
          previous=previous, emit=emit)
    approve(call.invocation)
    (trajectory,), (scored,) = measure("inference", lambda: sampled(runtime, (call.request,), reference, identities=identities))
    result(call, trajectory, identities=identities, emit=emit, reference=scored)
