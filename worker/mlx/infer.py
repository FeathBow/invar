import argparse
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
import sys

from worker.invocation import approve
from worker.hf.infer import arguments, run as standalone
from worker.mlx import inference as mlx_inference
from worker.mlx.metrics import measure
from worker.mlx import model as mlx_model
from worker.resident import Transcript


def run(options, *, loader=mlx_model.load, protocol=standalone):
    output = sys.stdout
    configuration = mlx_model.configuration(options.config)
    with redirect_stdout(sys.stderr), ExitStack() as scope:
        transcript = Transcript(output)
        measured = partial(measure, emit=transcript.emit)

        def load(cache, adapter, *, expected):
            return loader(cache, scope=scope, configuration=configuration, measure=measured,
                          emit=transcript.emit, initial=(adapter, expected))

        protocol(options, source=sys.stdin, loader=load, permission=approve,
              execute=partial(mlx_inference.execute, measure=measured, emit=transcript.emit, sampling=configuration.sampling()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    run(arguments(parser=parser))


if __name__ == "__main__":
    main()
