import argparse
from pathlib import Path
import sys

from worker.invocation import approve
from worker.session import declare, serve


def main():
    from functools import partial
    from worker.hf.inference import execute, load
    from worker.hf.metrics import measure
    from worker.hf.model import declared

    parser = declare(argparse.ArgumentParser(description="Execute a bound inference batch with one model load"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    options = parser.parse_args()
    serve(options, source=sys.stdin, loader=partial(load, declared=declared(options.config)),
          execute=partial(execute, measure=measure), permission=approve)


if __name__ == "__main__":
    main()
