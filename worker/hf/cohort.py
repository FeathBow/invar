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

    expected = calls[0].identities
    if any(call.identities != expected for call in calls):
        raise ValueError("Resident group members require the same materialization")
    identities = materialized(runtime, expected)
    readiness = [capture(partial(ready, call, identities, model=runtime.identity, previous=None)) for call in calls]
    emit("consumed", {"format": FORMAT, "calls": readiness})
    approve(tuple(call.invocation for call in calls))
    trajectories, scores = measure("inference", lambda: sampled(runtime, tuple(call.request for call in calls), reference,
                                                                identities=identities))
    completed = [capture(partial(result, call, trajectory, identities=identities, reference=scored))
                 for call, trajectory, scored in zip(calls, trajectories, scores, strict=True)]
    emit("result", {"format": FORMAT, "calls": completed})


def activate(runtime, adapter, *, expected):
    from worker.hf.policy import activate as install, read_adapter
    from worker.implementation import INFERENCE

    install(runtime.model, read_adapter(adapter, expected["adapter"]), base=expected["base"], assembly=expected["assembly"],
            role=INFERENCE)
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

    parser = argparse.ArgumentParser(description="Core-owned resident Hugging Face inference")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    run(parser.parse_args(), load=load, measure=measure)


if __name__ == "__main__":
    main()
