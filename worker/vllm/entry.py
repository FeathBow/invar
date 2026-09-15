import argparse
from contextlib import ExitStack, redirect_stdout
from functools import partial
import json
from pathlib import Path
import sys


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", type=Path, required=True)
    return result


def batch_arguments(description):
    result = parser(description)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--adapter", type=Path, required=True)
    return result.parse_args()


def report(stream, stage, values):
    print(json.dumps({"stage": stage, **values}, allow_nan=False), file=stream, flush=True)


def factory(**arguments):
    from vllm import LLM
    from worker.vllm.gdn import install

    install()
    return LLM(**arguments)


def components(options, stack, *, emit, engine_factory=factory):
    from huggingface_hub import snapshot_download
    from vllm.lora.request import LoRARequest
    from worker.hf.metrics import measure
    from worker.hf.operation import load as tokenizer
    from worker.vllm.configuration import read
    from worker.vllm.runtime import bound, execute, load

    timed = partial(measure, emit=emit)
    configured = partial(load, config=read(options.config), factory=engine_factory, tokenizer_loader=tokenizer,
                         resolve=snapshot_download, selection_factory=LoRARequest, measure=timed, emit=emit)

    def loader(cache, adapter, *, expected):
        runtime = stack.enter_context(configured(cache, adapter))
        bound(dict(runtime.identities), expected)
        return runtime

    return loader, partial(execute, measure=timed, emit=emit), configured


def run(options, *, protocol, engine_factory=factory):
    output = sys.stdout
    with redirect_stdout(sys.stderr), ExitStack() as stack:
        from worker.invocation import approve

        loader, execute, _ = components(options, stack, emit=partial(report, output), engine_factory=engine_factory)
        protocol(options, source=sys.stdin, loader=loader, execute=execute,
                 permission=partial(approve, source=sys.stdin))


def run_batch(options):
    output = sys.stdout
    with redirect_stdout(sys.stderr), ExitStack() as stack:
        from worker.batch import approve, serve
        from worker.hf.metrics import measure
        from worker.vllm.runtime import execute_batch

        emit = partial(report, output)
        loader, _, _ = components(options, stack, emit=emit)
        serve(options, source=sys.stdin, loader=loader,
              execute=partial(execute_batch, measure=partial(measure, emit=emit), emit=emit),
              permission=partial(approve, source=sys.stdin))


def inspect(options, *, engine_factory=factory):
    output = sys.stdout
    with redirect_stdout(sys.stderr), ExitStack() as stack:
        _, _, configured = components(options, stack, emit=partial(report, output), engine_factory=engine_factory)
        runtime = stack.enter_context(configured(options.cache, options.adapter))
        report(output, "identified", {"model": runtime.identity[0], "revision": runtime.identity[1],
                                      "handoff": runtime.receipt, "source": dict(runtime.source),
                                      **dict(runtime.identities)})
