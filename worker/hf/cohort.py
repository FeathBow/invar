import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from functools import partial
from pathlib import Path
import sys

from worker.batch import FORMAT, capture
from worker.report import ready, result


def execute_batch(runtime, calls, *, approve, measure, emit, reference=None):
    from worker.hf.inference import materialized, sampled
    from worker.hf.operation import described

    expected = calls[0].identities
    if any(call.identities != expected for call in calls):
        raise ValueError("Resident group members require the same materialization")
    identities = materialized(runtime, expected)
    description = described(runtime.tokenizer, identities["tokenizer"])
    readiness = [capture(partial(ready, call, identities, model=runtime.identity, description=description, previous=None))
                 for call in calls]
    emit("consumed", {"format": FORMAT, "calls": readiness})
    approve(tuple(call.invocation for call in calls))
    trajectories, scores = measure("inference", lambda: sampled(runtime, tuple(call.request for call in calls), reference,
                                                                identities=identities))
    completed = [capture(partial(result, call, trajectory, identities=identities, reference=scored))
                 for call, trajectory, scored in zip(calls, trajectories, scores, strict=True)]
    emit("result", {"format": FORMAT, "calls": completed})


def activate(runtime, adapter, *, expected):
    from worker.hf.inference import installed
    from worker.hf.policy import read_adapter

    installed(runtime.model, read_adapter(adapter, expected["adapter"]), expected)
    return replace(runtime, adapter=adapter)


def run(options, *, load, measure):
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        from worker.dispatch import serve
        from worker.resident import Owner, Transcript

        transcript = Transcript(output)
        serve(Owner(role="inference", session=options.session), source=sys.stdin, transcript=transcript,
              load=partial(load, options.cache, emit=transcript.emit), activate=activate,
              execute=partial(execute_batch, measure=partial(measure, emit=transcript.emit), emit=transcript.emit),
              release=lambda runtime: None, close=lambda: None, measure=measure)


def main():
    from worker.hf.inference import load
    from worker.hf.metrics import measure
    from worker.hf.model import declared

    parser = argparse.ArgumentParser(description="Core-owned resident Hugging Face inference")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    options = parser.parse_args()
    run(options, load=partial(load, declared=declared(options.config)), measure=measure)


if __name__ == "__main__":
    main()
