import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import argparse
from contextlib import redirect_stdout
from functools import partial
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from worker.hf import assembly, frozen
from worker.hf.checkpoint import ADAM_BETAS, ADAM_EPSILON, LEARNING_RATE, checkpoint
from worker.hf.inference import Runtime, execute
from worker.hf.learning import parameters
from worker.hf.metrics import report
from worker.hf.operation import verify
from worker.hf.policy import activate, read_adapter
from worker.hf.step import file_digest
from worker.hf.tensors import digest
from worker.implementation import INFERENCE
from worker.invocation import approve
from worker.tests.hf.inference import IDENTITY, TEST_THREADS, measured, model
from worker.tests.hf.tokenization import make_tokenizer


def initialize(output):
    learner, tokenizer = model(), make_tokenizer()
    optimizer = torch.optim.AdamW(parameters(learner), lr=LEARNING_RATE, betas=ADAM_BETAS,
                                  eps=ADAM_EPSILON, weight_decay=0.0, foreach=False, fused=False)
    output.mkdir(exist_ok=False)
    saved = checkpoint(learner, optimizer, output, tokenizer=tokenizer, expected=None)
    materialized = torch.load(output / "learner.pt", weights_only=True)
    report("initial", {"policy": digest(saved), "learner": file_digest(output / "learner.pt"),
                       "tokenizer": verify(tokenizer, materialized["tokenizer"]),
                       "base": materialized["base"], "assembly": materialized["assembly"],
                       "inference": {"base": frozen.digest(learner), "assembly": assembly.digest(learner, INFERENCE)}})


def inference(cache, adapter, *, expected, emit=None):
    from worker.tests.hf.handshake import cpu_measure

    state = read_adapter(adapter, expected["adapter"])
    tokenizer = make_tokenizer()
    verify(tokenizer, expected["tokenizer"])
    loaded = measured("load", model) if emit is None else cpu_measure("load", model, emit=emit)
    activate(loaded, state, base=expected["base"], assembly=expected["assembly"], role=INFERENCE)
    return Runtime(model=loaded, tokenizer=tokenizer, adapter=adapter, device="cpu", identity=IDENTITY)


def learn(options):
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        from worker.hf.probability import logprobs
        from worker.hf.runtime import close
        from worker.learner import serve
        from worker.resident import Owner, Transcript
        from worker.tests.hf.handshake import cpu_measure

        transcript = Transcript(output)

        def loader(options, request):
            return cpu_measure("load", model, emit=transcript.emit), make_tokenizer(), IDENTITY

        serve(Owner(role="learning", session=options.session), options, source=sys.stdin, transcript=transcript,
              loader=loader, measure=cpu_measure, evaluate=partial(logprobs, device="cpu"), close=close)


def main():
    parser = argparse.ArgumentParser(description="Hugging Face test model entry for end-to-end core tests")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--reference")
    parser.add_argument("--reference-digest")
    parser.add_argument("--session", type=int)
    parser.add_argument("--digest")
    parser.add_argument("--tokenizer-digest")
    parser.add_argument("--base-digest")
    parser.add_argument("--assembly-digest")
    parser.add_argument("--prompt")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    options = parser.parse_args()
    torch.set_num_threads(TEST_THREADS)
    if options.output is not None:
        initialize(options.output)
    elif options.session is not None and options.reference is not None:
        learn(options)
    elif options.session is not None:
        from worker.hf.cohort import run
        from worker.tests.hf.handshake import cpu_measure

        run(options, load=inference, measure=cpu_measure)
    elif options.digest is not None:
        from worker.hf.infer import run

        run(options, source=sys.stdin, loader=inference, execute=partial(execute, measure=measured), permission=approve)
    else:
        from worker.session import serve

        serve(options, source=sys.stdin, loader=inference, execute=partial(execute, measure=measured), permission=approve)


if __name__ == "__main__":
    main()
