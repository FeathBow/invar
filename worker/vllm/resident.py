from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
import sys

from worker.vllm.entry import parser


def run(options):
    output = sys.stdout
    with redirect_stdout(sys.stderr), ExitStack() as stack:
        from vllm.lora.request import LoRARequest
        from worker.dispatch import serve
        from worker.hf.metrics import measure
        from worker.resident import Owner, Transcript
        from worker.vllm.configuration import read
        from worker.vllm.entry import components
        from worker.vllm.residency import release, scheduler, select
        from worker.vllm.runtime import execute_batch

        transcript = Transcript(output)
        loader, _, _ = components(options, stack, emit=transcript.emit)

        def load(adapter, *, expected):
            runtime = loader(options.cache, adapter, expected=expected)
            scheduler(runtime)
            return runtime

        serve(Owner(role="inference", session=options.session), source=sys.stdin, transcript=transcript,
              load=load,
              activate=partial(select, config=read(options.config), selection_factory=LoRARequest),
              execute=partial(execute_batch, measure=partial(measure, emit=transcript.emit), emit=transcript.emit,
                              config=read(options.config), selection_factory=LoRARequest),
              release=release, close=stack.close, measure=measure)


def main():
    arguments = parser("Core-owned resident native inference")
    arguments.add_argument("--cache", type=Path, required=True)
    arguments.add_argument("--session", type=int, required=True)
    run(arguments.parse_args())


if __name__ == "__main__":
    main()
