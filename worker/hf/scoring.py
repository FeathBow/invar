import argparse
from contextlib import redirect_stdout
from functools import partial
import hashlib
from pathlib import Path
import sys

from worker import core, registry, resident
from worker.hf.inference import load, materialized
from worker.hf.metrics import measure
from worker.hf.model import declared
from worker.hf.score import forced
from worker.invocation import approve, request as request_value
from worker.resident import Owner, Transcript
from worker.scoring import decode, prescribed, provenance
from worker.tokenization import prompt

FORMAT = "invar-cached-path-score-v1"
ENGINE = "transformers.forced_path"
CACHE_ORIGIN = "fresh model cache per path; no source cache input"
NUMERICS = "transformers/float32_softmax"
IMPLEMENTATIONS = ("worker.scoring", "worker.hf.scoring", "worker.hf.score", "worker.hf.rollout", "worker.hf.decoding",
                   "worker.hf.inference", "worker.hf.model", "worker.hf.policy", "worker.tokenization")


def observe(runtime, selected, identities, *, measured):
    path = selected.path
    prescribed(selected.request, path, runtime.tokenizer, tuple(prompt(runtime.tokenizer, selected.request.prompt)[0].tolist()))
    actual = measured("verify_before", lambda: materialized(runtime, identities))
    words = measured("cross_score", lambda: forced(runtime.model, selected, device=runtime.device))
    measured("verify_after", lambda: materialized(runtime, identities))
    return {"format": FORMAT, "role": "cached_behavior_cross_score", "use_admission": "not_evaluated",
            "source_inspection_sha256": hashlib.sha256(selected.inspection).hexdigest(),
            "source": core.decode(selected.inspection),
            "target": {**actual, "model": runtime.identity[0], "revision": runtime.identity[1], "numerics": NUMERICS},
            "request": request_value(selected.request), "prefix_tokens": list(path.prefix),
            "response_tokens": list(path.response), "log_probability_bits": words,
            "probability": {"role": "behavior", "log_base": "e", "representation": "F32 words",
                            "zero_support_word": 0xff800000, "temperature": selected.request.temperature,
                            "mask": "none", "top_k": "disabled", "top_p": "disabled"},
            "execution": {"engine": ENGINE, "cache_origin": CACHE_ORIGIN, "truncated": selected.truncated},
            "implementation": provenance(IMPLEMENTATIONS, packages=("torch", "transformers", "peft"))}


def run(options, *, loader=load, measure=measure, source=None, output=None):
    incoming = sys.stdin if source is None else source
    transcript = Transcript(sys.stdout if output is None else output)
    with redirect_stdout(sys.stderr):
        if options.session is not None:
            serve(Owner(role="teacher", session=options.session), options, source=incoming, transcript=transcript,
                  loader=loader, measure=measure)
            return
        envelope = core.decode(incoming.readline())
        call, _, _ = decode(envelope)
        runtime = loader(options.cache, options.adapter, expected=call.identities, emit=transcript.emit, declared=declared(options.config))
        execute(runtime, envelope, source=incoming, transcript=transcript, measure=measure)


def serve(owner, options, *, source, transcript, loader, measure):
    runtime = None
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
        if runtime is None:
            runtime = loader(options.cache, options.adapter, expected=call.identities, emit=transcript.emit, declared=declared(options.config))
        execute(runtime, value, source=source, transcript=transcript, measure=measure)
        resident.release(owner, (call.load,), source=source, transcript=transcript, operation=lambda: None, measure=measure)
        history = following


def execute(runtime, envelope, *, source, transcript, measure):
    call, selected, _ = decode(envelope)
    actual = materialized(runtime, call.identities)
    transcript.emit("loaded_adapter", {"binding": call.invocation.binding(),
                    "load": registry.invocation(call.load), "image": registry.image(actual),
                    "requested": call.identities["adapter"], "consumed": actual["adapter"],
                    **{key: actual[key] for key in ("tokenizer", "base", "assembly")},
                    "model": runtime.identity[0], "revision": runtime.identity[1]})
    transcript.emit("consumed", {**envelope, **actual, "request": request_value(call.request)})
    approve(call.invocation, source=source)
    observation = observe(runtime, selected, call.identities, measured=partial(measure, emit=transcript.emit))
    transcript.emit("score_result", {"binding": call.invocation.binding(), "observation": observation})


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--session", type=int)
    return parser.parse_args()


def main():
    run(arguments())


if __name__ == "__main__":
    main()
