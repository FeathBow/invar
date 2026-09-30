import argparse
from pathlib import Path
import sys

from worker.invocation import approve
from worker.session import declare, serve


def main():
    from functools import partial
    from worker.hf.inference import execute, load
    from worker.hf.metrics import measure

    parser = declare(argparse.ArgumentParser(description="Execute a bound inference batch with one model load"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    serve(parser.parse_args(), source=sys.stdin, loader=load,
          execute=partial(execute, measure=measure), permission=approve)


if __name__ == "__main__":
    main()
