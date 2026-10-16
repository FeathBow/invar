import argparse
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
import sys
import tempfile

from worker import core, probeoutput, registry, resident
from worker.invocation import approve, request as request_value
from worker.mlx import model as mlx_model
from worker.mlx import score as diagnostic
from worker.mlx import tokenization as mlx_tokenization
from worker.mlx.crossscore import coordinates
from worker.mlx.metrics import measure
from worker.resident import Owner, Transcript
from worker.probestore import Store
from worker.scoring import decode
from worker.implementation import INFERENCE


def run(options, *, loader=mlx_model.load, source=None, output=None):
    incoming = sys.stdin if source is None else source
    transcript = Transcript(sys.stdout if output is None else output)
    configuration = mlx_model.configuration(options.config)
    with redirect_stdout(sys.stderr), ExitStack() as scope:
        if options.session is not None:
            serve(Owner(role="teacher", session=options.session), options, configuration, source=incoming, transcript=transcript,
                  loader=loader, scope=scope)
            return
        envelope = core.decode(incoming.readline())
        call, _, _ = decode(envelope)
        measurements, emit = recorder(transcript)
        loaded = loader(options.cache, scope=scope, configuration=configuration, measure=partial(measure, emit=emit),
                        emit=emit, initial=(options.adapter, call.identities))
        execute(loaded, envelope, configuration, source=incoming, transcript=transcript, measurements=measurements, emit=emit)


def serve(owner, options, configuration, *, source, transcript, loader, scope):
    loaded = None
    history = resident.History()
    while True:
        value = resident.decode(source.readline())
        if isinstance(value, dict) and value.get("format") == resident.FORMAT:
            resident.closing(owner, value)
            resident.close(owner, history.groups, source=source, transcript=transcript, operation=lambda: None, measure=measure)
            return
        call, _, _ = decode(value)
        following = history.advance((call,))
        transcript.begin()
        measurements, emit = recorder(transcript)
        if loaded is None:
            loaded = loader(options.cache, scope=scope, configuration=configuration, measure=partial(measure, emit=emit),
                            emit=emit, initial=(options.adapter, call.identities))
        execute(loaded, value, configuration, source=source, transcript=transcript, measurements=measurements, emit=emit)
        resident.release(owner, (call.load,), source=source, transcript=transcript, operation=lambda: None, measure=measure)
        history = following


def recorder(transcript):
    measurements = []

    def emit(stage, values):
        measurements.append({"stage": stage, **values})
        transcript.emit(stage, values)

    return measurements, emit


def execute(loaded, envelope, configuration, *, source, transcript, measurements, emit):
    call, selected, probe = decode(envelope)
    measured = partial(measure, emit=emit)
    actual = mlx_model.verify(loaded, call.identities, INFERENCE)
    coordinated = coordinates(loaded.model)
    transcript.emit("loaded_adapter", {"binding": call.invocation.binding(),
                    "load": registry.invocation(call.load), "image": registry.image(actual),
                    "requested": call.identities["adapter"], "consumed": actual["adapter"],
                    **{key: actual[key] for key in ("tokenizer", "base", "assembly")},
                    "tokenizer_description": mlx_tokenization.described(loaded.tokenizer, actual["tokenizer"]),
                    "coordinates": coordinated, "model": loaded.identity[0], "revision": loaded.identity[1]})
    transcript.emit("consumed", {**envelope, **actual, "request": request_value(call.request)})
    approve(call.invocation, source=source)
    with ExitStack() as scope:
        store = None if probe is None else Store(scope.enter_context(tempfile.TemporaryFile()), probe=probe)
        result = diagnostic.observe(loaded, selected, expected=actual, coordinated=coordinated,
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
    parser.add_argument("--session", type=int)
    return parser.parse_args()


def main():
    run(arguments())


if __name__ == "__main__":
    main()
