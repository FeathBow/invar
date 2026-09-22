import argparse
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
import sys
import tempfile

from worker import core, probeoutput, registry
from worker.invocation import approve, request as request_value
from worker.mlx import model as mlx_model
from worker.mlx import score as diagnostic
from worker.mlx.metrics import measure
from worker.resident import Transcript
from worker.probestore import Store
from worker.scoring import decode


def run(options, *, loader=mlx_model.load, source=None, output=None):
    incoming = sys.stdin if source is None else source
    transcript = Transcript(sys.stdout if output is None else output)
    envelope = core.decode(incoming.readline())
    call, selected, probe = decode(envelope)
    configuration = mlx_model.configuration(options.config)
    measurements = []

    def emit(stage, values):
        measurements.append({"stage": stage, **values})
        transcript.emit(stage, values)

    measured = partial(measure, emit=emit)
    with redirect_stdout(sys.stderr), ExitStack() as scope:
        loaded = loader(options.cache, scope=scope, configuration=configuration, measure=measured,
                        emit=emit, initial=(options.adapter, call.identities))
        actual = mlx_model.verify(loaded, call.identities)
        transcript.emit("loaded_adapter", {"binding": call.invocation.binding(),
                        "load": registry.invocation(call.load), "image": registry.image(actual),
                        "requested": call.identities["adapter"], "consumed": actual["adapter"],
                        **{key: actual[key] for key in ("tokenizer", "base", "assembly")},
                        "model": loaded.identity[0], "revision": loaded.identity[1]})
        transcript.emit("consumed", {**envelope, **actual, "request": request_value(call.request)})
        approve(call.invocation, source=incoming)
        store = None if probe is None else Store(scope.enter_context(tempfile.TemporaryFile()), probe=probe)
        result = diagnostic.observe(loaded, selected, expected=actual,
                                    sampling=configuration.sampling(), measured=measured, probe=probe, store=store)
        if probe is not None:
            result = {**result, "measurements": measurements}
        provenance = diagnostic.implementation((*result["implementation"]["sources_sha256"], __name__))
        values = {"binding": call.invocation.binding(), "observation": {**result, "implementation": provenance}}
        if probe is None:
            transcript.emit("score_result", values)
        else:
            transcript.emit_stream("score_result", values, encode=probeoutput.chunks)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main():
    run(arguments())


if __name__ == "__main__":
    main()
