import argparse
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
from functools import partial
import gc
from pathlib import Path
import sys

import mlx.core as mx

from worker.dispatch import serve
from worker.mlx import inference as mlx_inference
from worker.mlx.metrics import measure
from worker.mlx import model as mlx_model
from worker.resident import Owner, Transcript
from worker.implementation import INFERENCE


@dataclass(frozen=True, kw_only=True)
class Selected:
    loaded: mlx_model.Loaded
    identities: dict


def activate(runtime, path, *, expected):
    mlx_model.verify(runtime.loaded, runtime.identities, INFERENCE)
    if expected != runtime.identities:
        mlx_model.activate(runtime.loaded, path, expected=expected)
    return Selected(loaded=runtime.loaded, identities=expected)


def execute(runtime, calls, **operations):
    return mlx_inference.execute_batch(runtime.loaded, calls, **operations)


def release(_runtime=None):
    gc.collect()
    mx.clear_cache()


def close(scope):
    scope.close()
    release()


def run(options, *, loader=mlx_model.load):
    output = sys.stdout
    configuration = mlx_model.configuration(options.config)
    with redirect_stdout(sys.stderr), ExitStack() as scope:
        transcript = Transcript(output)
        measured = partial(measure, emit=transcript.emit)

        def load(adapter, *, expected):
            loaded = loader(options.cache, scope=scope, configuration=configuration, measure=measured,
                            emit=transcript.emit, initial=(adapter, expected))
            return Selected(loaded=loaded, identities=expected)

        serve(Owner(role="inference", session=options.session), source=sys.stdin, transcript=transcript,
              load=load, activate=activate,
              execute=partial(execute, measure=measured, emit=transcript.emit,
                              sampling=configuration.sampling()),
              release=release, close=partial(close, scope), measure=measure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--config", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
