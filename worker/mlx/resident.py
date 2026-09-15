import argparse
from contextlib import ExitStack, redirect_stdout
from functools import partial
import gc
from pathlib import Path
import sys

import mlx.core as mx

from worker import batch as inference_batch
from worker.invocation import approve
from worker import inputs as learning_resident_input
from worker.mlx import inference as mlx_inference
from worker.mlx import learner as mlx_learner
from worker.mlx.metrics import measure
from worker.mlx import model as mlx_model
from worker.mlx import tensors as mlx_tensors
from worker import resident


def inference(loaded, learner, group):
    expected = group.calls[0].identities
    if any(call.identities != expected for call in group.calls):
        raise ValueError("Shared inference members require the same materialization")
    if learner is None:
        return mlx_model.activate(loaded, group.adapter, expected=expected)
    actual = mlx_learner.verify(learner)
    if expected != actual:
        raise ValueError("Shared inference must consume the completed native successor")
    mlx_tensors.policy(group.adapter, expected["adapter"])
    return loaded


def released(learner):
    if learner is not None:
        mlx_learner.release(learner)
    else:
        gc.collect()
        mx.clear_cache()


def serve(options, *, loader, scope, source, transcript):
    owner = resident.Owner(role="shared", session=options.session)
    configuration = mlx_model.configuration(options.config)
    measured = partial(measure, emit=transcript.emit)
    loaded, learner = None, None
    history = resident.History()

    def load(initial=None):
        return loader(options.cache, scope=scope, configuration=configuration, measure=measured,
                      emit=transcript.emit, initial=initial)

    def shutdown():
        nonlocal loaded, learner
        loaded, learner = None, None
        scope.close()
        gc.collect()
        mx.clear_cache()

    while True:
        raw = source.readline()
        value = resident.decode(raw)
        if isinstance(value, dict) and value.get("format") == resident.FORMAT:
            resident.closing(owner, value)
            resident.close(owner, history.groups, source=source, transcript=transcript, operation=shutdown, measure=measure)
            return
        transcript.begin()
        if isinstance(value, dict) and value.get("format") == inference_batch.FORMAT:
            group = inference_batch.decode(raw)
            next_history = history.advance(group.calls)
            if loaded is None:
                loaded = load((group.adapter, group.calls[0].identities))
            else:
                loaded = measured("activation", partial(inference, loaded, learner, group))
            mlx_inference.execute_batch(loaded, group.calls, approve=partial(inference_batch.approve, source=source),
                                        measure=measured, emit=transcript.emit, sampling=configuration.sampling())
            loads = tuple(call.load for call in group.calls)
        else:
            call, paths = learning_resident_input.decode(value, options)
            next_history = history.advance((call,))
            paths.output.mkdir(exist_ok=False)
            if loaded is None:
                loaded = load()
            if learner is None:
                learner = measured("activation", partial(mlx_learner.restore, loaded, call.request, paths))
            else:
                learner = measured("activation", partial(mlx_learner.activate, learner, call.request, paths))
            learner = mlx_learner.execute(learner, call, paths.output, measure=measured,
                                          approve=partial(approve, source=source), emit=transcript.emit)
            loads = (call.load,)
        resident.release(owner, loads, source=source, transcript=transcript,
                         operation=partial(released, learner), measure=measure)
        history = next_history


def run(options, *, loader=mlx_model.load):
    if not options.shared:
        raise ValueError("This worker requires explicit shared numerical ownership")
    output = sys.stdout
    with redirect_stdout(sys.stderr), ExitStack() as scope:
        serve(options, loader=loader, scope=scope, source=sys.stdin, transcript=resident.Transcript(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--config", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
