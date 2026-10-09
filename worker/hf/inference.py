from dataclasses import dataclass

import torch

from worker.report import ready, result
from worker.hf.policy import activate, read_adapter, verify as verify_adapter
from worker.hf.model import CPU_THREADS, DEFAULT_SEED, MODEL, REVISION, load_model
from worker.hf.metrics import measure, report
from worker.hf.score import referenced
from worker.hf.rollout import generate
from worker.implementation import INFERENCE
from worker.hf.operation import load as load_tokenizer
from worker.hf.operation import verify

@dataclass(frozen=True, kw_only=True)
class Runtime:
    model: torch.nn.Module
    tokenizer: object
    adapter: object
    device: str
    identity: tuple[str, str]


def load(cache, adapter, *, expected, emit=report):
    from huggingface_hub import snapshot_download

    state = read_adapter(adapter, expected["adapter"])
    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=cache, local_files_only=True)
    tokenizer = load_tokenizer(path)
    verify(tokenizer, expected["tokenizer"])
    model = measure("load", lambda: load_model(path, role=INFERENCE, emit=emit), emit=emit)
    activate(model, state, base=expected["base"], assembly=expected["assembly"], role=INFERENCE)
    return Runtime(model=model, tokenizer=tokenizer, adapter=adapter, device="cuda", identity=(MODEL, REVISION))


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
    ready(call, identities, model=runtime.identity, previous=previous, emit=emit)
    approve(call.invocation)
    (trajectory,), (scored,) = measure("inference", lambda: sampled(runtime, (call.request,), reference, identities=identities))
    result(call, trajectory, identities=identities, emit=emit, reference=scored)
