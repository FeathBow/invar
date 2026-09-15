import argparse
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
import sys


def run(options):
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        from worker.learner import serve
        from worker.hf.runtime import close
        from worker.hf.metrics import measure
        from worker.resident import Owner, Transcript
        from worker.hf.rollout import logprobs
        from worker.hf.step import load_with

        transcript = Transcript(output)
        loader = partial(load_with, measure=partial(measure, emit=transcript.emit), emit=transcript.emit)
        serve(Owner(role="learning", session=options.session), options, source=sys.stdin,
              transcript=transcript, loader=loader, measure=measure, evaluate=logprobs, close=close)


def main():
    parser = argparse.ArgumentParser(description="Execute core-owned resident GRPO updates and stage each successor")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
