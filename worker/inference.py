from dataclasses import dataclass

import torch
import registry

from invocation import request as reported_request
from policy import activate, read_adapter
from probe import CPU_THREADS, DEFAULT_SEED, MODEL, REVISION, load_model, measure, report
from rollout import generate
from operation import load as load_tokenizer
from operation import verify

FP32_MASK = (1 << 32) - 1


@dataclass(frozen=True, kw_only=True)
class Runtime:
    model: torch.nn.Module
    tokenizer: object
    adapter: object
    device: str
    identity: tuple[str, str]


def load(cache, adapter, *, expected):
    from huggingface_hub import snapshot_download

    read_adapter(adapter, expected["adapter"])
    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=cache, local_files_only=True)
    tokenizer = load_tokenizer(path)
    verify(tokenizer, expected["tokenizer"])
    model = measure("load", lambda: load_model(path))
    return Runtime(model=model, tokenizer=tokenizer, adapter=adapter, device="cuda", identity=(MODEL, REVISION))


def execute(runtime, call, *, approve, measure, previous=None):
    requested = call.identities
    state = read_adapter(runtime.adapter, requested["adapter"])
    tokenizer = verify(runtime.tokenizer, requested["tokenizer"])
    consumed = activate(runtime.model, state, base=requested["base"], assembly=requested["assembly"])
    materialization = {"tokenizer": tokenizer, "base": requested["base"], "assembly": requested["assembly"]}
    if previous is not None:
        report('unloaded_adapter', registry.invocation(previous))
    report("loaded_adapter", {"binding": call.invocation.binding(), "requested": requested["adapter"],
                              "consumed": consumed, **materialization,
                              'load': registry.invocation(call.load),
                              'image': registry.image({'adapter': consumed, **materialization}),
                              "model": runtime.identity[0], "revision": runtime.identity[1],
                              "scope": "tensor binding; not publication or numerical certification"})
    report("consumed", {"binding": call.invocation.binding(), "program": call.invocation.program,
                        'load': registry.invocation(call.load),
                        "adapter": consumed, **materialization, "request": reported_request(call.request)})
    approve(call.invocation)
    trajectory = measure("inference", lambda: generate(runtime.model, runtime.tokenizer, call.request,
                                                        device=runtime.device))
    report("result", {"binding": call.invocation.binding(), "adapter": consumed, **materialization,
                      "request": reported_request(trajectory.request),
                      "tokens": trajectory.tokens[0].tolist(), "prompt_length": trajectory.prompt_length,
                      "behavior": trajectory.behavior.tolist(), "text": trajectory.text,
                      "behavior_bits": [value & FP32_MASK for value in trajectory.behavior.view(torch.int32).tolist()],
                      "truncated": trajectory.truncated})
