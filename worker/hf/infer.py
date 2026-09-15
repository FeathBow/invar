import argparse
import math
import json
import sys
from pathlib import Path

from worker.cohort import fields, identity


def arguments(*, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Standalone inference with an explicitly bound adapter")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--digest", required=True, help="Expected canonical adapter tensor SHA-256")
    parser.add_argument("--tokenizer-digest", required=True, help="Expected tokenizer operation SHA-256")
    parser.add_argument("--base-digest", required=True, help="Expected frozen model tensor SHA-256")
    parser.add_argument("--assembly-digest", required=True, help="Expected model assembly SHA-256")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    options = parser.parse_args()
    for value in (options.digest, options.tokenizer_digest, options.base_digest, options.assembly_digest):
        identity(value)
    if options.tokens <= 0 or not math.isfinite(options.temperature) or options.temperature <= 0:
        parser.error("--tokens and --temperature must be finite and positive")
    return options


def run(options, *, source, loader, execute, permission):
    from worker.hf.session import decode, unique

    invocation = json.loads(source.readline(), object_pairs_hook=unique)
    fields(invocation, 'binding program load')
    call = decode({**invocation,
                   "adapter": options.digest, "tokenizer": options.tokenizer_digest,
                   "base": options.base_digest, "assembly": options.assembly_digest,
                   "request": {"prompt": options.prompt, "tokens": options.tokens,
                               "temperature": options.temperature, "seed": options.seed}})
    runtime = loader(options.cache, options.adapter, expected=call.identities)
    execute(runtime, call, approve=permission)


def main():
    options = arguments()
    from functools import partial
    from worker.hf.inference import execute, load
    from worker.invocation import approve
    from worker.hf.metrics import measure

    run(options, source=sys.stdin, loader=load, execute=partial(execute, measure=measure), permission=approve)


if __name__ == "__main__":
    main()
