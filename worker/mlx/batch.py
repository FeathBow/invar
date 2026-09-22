import argparse
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
import sys

from worker import batch
from worker.mlx import inference, model
from worker.mlx.metrics import measure
from worker.resident import Transcript


def run(options, *, loader=model.load):
    output = sys.stdout
    configuration = model.configuration(options.config)
    with redirect_stdout(sys.stderr), ExitStack() as scope:
        transcript = Transcript(output)
        measured = partial(measure, emit=transcript.emit)

        def load(cache, adapter, *, expected):
            return loader(cache, scope=scope, configuration=configuration, measure=measured,
                          emit=transcript.emit, initial=(adapter, expected))

        batch.serve(options, source=sys.stdin, loader=load,
                    execute=partial(inference.execute_batch, measure=measured,
                                    emit=transcript.emit, sampling=configuration.sampling()),
                    permission=partial(batch.approve, source=sys.stdin))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main():
    run(arguments())


if __name__ == "__main__":
    main()
