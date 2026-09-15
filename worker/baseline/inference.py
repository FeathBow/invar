import argparse
from contextlib import redirect_stdout
from functools import partial
import json
from pathlib import Path
import sys

from worker.cohort import fields, identity
from worker import core
from worker.baseline import data as product_data


def report(stream, stage, values):
    print(json.dumps({"stage": stage, **values}, allow_nan=False), file=stream, flush=True)


def activate(runtime, directory, *, policy, config):
    from vllm.lora.request import LoRARequest
    from worker.baseline.handoff import activate as install

    return install(runtime, Path(directory), policy=identity(policy), config=config, selection_factory=LoRARequest)


def generate(runtime, tasks):
    from worker.vllm.rollout import generate
    from worker.vllm.residency import available, released

    actual = generate(runtime.engine, runtime.tokenizer, product_data.requests(tasks),
                      loras=[runtime.lora] * len(tasks))
    available(runtime)
    if runtime.engine.llm_engine.step():
        raise RuntimeError("Native engine returned extra completion output")
    runtime.engine.collective_rpc(released, kwargs={"adapter_id": runtime.lora.lora_int_id})
    return [product_data.observation(value, dict(runtime.identities)) for value in actual]


def serve(runtime, config, *, source, measure, emit):
    for raw in source:
        value = core.decode(raw)
        if not isinstance(value, dict):
            raise ValueError("Expected a native composition operation object")
        operation = value.get("operation")
        if operation == "generate":
            fields(value, "operation tasks")
            if not isinstance(value["tasks"], list) or not value["tasks"]:
                raise ValueError("Expected a nonempty native task group")
            for task in value["tasks"]:
                fields(task, "name group prompt tokens temperature seed answer")
            actual = measure("inference", lambda: generate(runtime, value["tasks"]))
            emit("completed", {"results": actual})
        elif operation == "activate":
            fields(value, "operation adapter policy")
            runtime = measure("published_activation", lambda: activate(runtime, value["adapter"], policy=value["policy"], config=config))
            emit("completed", {"identities": dict(runtime.identities)})
        elif operation == "close":
            fields(value, "operation")
            emit("completed", {})
            return
        else:
            raise ValueError("Unknown native composition operation")
    raise ValueError("Native composition input ended without close")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "adapter", "config", "settings"):
        parser.add_argument("--" + name, type=Path, required=True)
    options = parser.parse_args()
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        from huggingface_hub import snapshot_download
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from worker.hf.metrics import measure
        from worker.hf.operation import load as tokenizer
        from worker.vllm.configuration import read
        from worker.vllm.runtime import load
        # The profile describes these modules even when ordinary native GDN and
        # generation kernels are selected. Importing does not install overrides.
        from worker.vllm import gdn as vllm_gdn

        emit = partial(report, output)
        timed = partial(measure, emit=emit)
        config = read(options.config)
        if config.template is None:
            raise ValueError("Native composition requires a checkpoint PEFT template")
        with load(options.cache, options.adapter, config=config, factory=LLM, tokenizer_loader=tokenizer,
                  resolve=snapshot_download, selection_factory=LoRARequest, measure=timed, emit=emit) as runtime:
            settings = core.decode(options.settings.read_text())
            expected = product_data.identities(settings, behavior=True)
            actual = dict(runtime.identities)
            if any(actual[name] != expected[name] for name in ("adapter", "tokenizer")):
                raise ValueError("Native product engine consumed a different initial policy or tokenizer")
            emit("ready", {"identities": actual, "model": runtime.identity[0], "revision": runtime.identity[1]})
            serve(runtime, config, source=sys.stdin, measure=timed, emit=emit)


if __name__ == "__main__":
    main()
