from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from functools import partial
import hashlib
import sys

from worker import core, probeoutput, registry
from worker.distribution import FP32_BYTES, PROBE_FORMAT
from worker.hf.metrics import measure
from worker.hf.operation import verify
from worker.invocation import approve, request as request_value
from worker.resident import Transcript
from worker.probeschema import NATIVE_CACHE, NATIVE_PATH, VLLM_ENGINE, native_relation
from worker.scoring import decode, provenance
from worker.vllm import entry
from worker.vllm.resources import Scoring

ENGINE = VLLM_ENGINE
CACHE_ORIGIN = NATIVE_CACHE
PATH_CONTROL = NATIVE_PATH
IMPLEMENTATIONS = ("worker.scoring", "worker.vllm.scoring", "worker.vllm.crossscore",
                   "worker.vllm.scoreobservation", "worker.vllm.probes", "worker.vllm.distribution",
                   "worker.distribution", "worker.probepacked", "worker.probeoutput", "worker.resident",
                   "worker.scalar", "worker.probeschema", "worker.vllm.resources", "torch.overrides",
                   "worker.vllm.prescribed", "worker.vllm.state", "worker.vllm.context", "worker.vllm.execution",
                   "worker.vllm.worker", "worker.vllm.runtime", "worker.vllm.entry",
                   "worker.vllm.profile", "worker.vllm.rollout", "worker.invocation",
                   "worker.hf.session", "vllm.v1.sample.sampler",
                   "vllm.v1.sample.ops.topk_topp_sampler", "vllm.model_executor.layers.batch_invariant",
                   "vllm.v1.worker.gpu_model_runner")


def observation(runtime, selected, *, scored, actual):
    return {"format": "invar-cached-path-score-v1", "role": "cached_behavior_cross_score",
            "use_admission": "not_evaluated",
            "source_inspection_sha256": hashlib.sha256(selected.inspection).hexdigest(),
            "source": core.decode(selected.inspection),
            "target": {**actual, "model": runtime.identity[0], "revision": runtime.identity[1],
                       "numerics": "vllm/processed_logprobs"},
            "request": request_value(selected.request), "prefix_tokens": list(scored.path.prefix),
            "response_tokens": list(scored.path.response), "log_probability_bits": list(scored.log_probability_bits),
            "probability": {"role": "behavior", "log_base": "e", "representation": "F32 words",
                            "zero_support_word": 0xff800000, "temperature": selected.request.temperature,
                            "mask": "none", "top_k": "disabled", "top_p": "disabled"},
            "execution": {"engine": ENGINE, "cache_origin": CACHE_ORIGIN, "path_control": PATH_CONTROL,
                          "native_sample_rows": scored.native_sample_rows,
                          "ignored_prefill_rows": scored.ignored_prefill_rows, "truncated": scored.truncated},
            "implementation": provenance(IMPLEMENTATIONS, packages=("torch", "vllm", "transformers"))}


def full_vocabulary(result, selected, *, observed, measurements, stream=False):
    if observed is None:
        return result
    vectors = {"steps": list(observed.probe.steps), "vocabulary": observed.vocabulary,
               "coordinates": "output token ids 0..vocabulary-1", "representation": "F32 probability words",
               "snapshots": (probeoutput.Rows(snapshots=observed.snapshots) if stream
                             else [asdict(value) for value in observed.snapshots]),
               "raw_payload_bytes": len(observed.snapshots) * observed.vocabulary * FP32_BYTES}
    return {**result, "format": PROBE_FORMAT, "role": "cached_behavior_full_vocabulary",
            "source_inspection": selected.inspection.decode("utf-8"), "full_vocabulary": vectors,
            "execution": {**result["execution"], "distribution": native_relation()},
            "measurements": list(measurements)}


def execute(runtime, envelope, *, incoming, transcript, measured, emit, measurements):
    from worker.vllm import crossscore, runtime as native

    call, selected, probe = decode(envelope)
    costs = Scoring(measured, emit=emit, probing=probe is not None)
    tokenizer = verify(runtime.tokenizer, call.identities["tokenizer"])
    if selected.truncated != (selected.path.response[-1] != runtime.tokenizer.eos_token_id):
        raise ValueError("Source stopping report differs from the actual target EOS")

    def verified(prepared):
        verify(runtime.tokenizer, tokenizer)
        actual = native.observed(prepared, runtime, tokenizer=tokenizer)
        native.bound(actual, call.identities)
        return actual

    def permission(prepared):
        actual = verified(prepared)
        transcript.emit("loaded_adapter", {"binding": call.invocation.binding(),
                        "load": registry.invocation(call.load), "image": registry.image(actual),
                        "requested": call.identities["adapter"], "consumed": actual["adapter"],
                        **{key: actual[key] for key in ("tokenizer", "base", "assembly")},
                        "model": runtime.identity[0], "revision": runtime.identity[1]})
        transcript.emit("consumed", envelope)
        approve(call.invocation, source=incoming)
        measured("verify_before", lambda: verified(prepared))

    execution = crossscore.score(runtime.engine, runtime.tokenizer, (call.request,), paths=(selected.path,),
                                   loras=(runtime.lora,), receipts=(runtime.receipt,), approve=permission,
                                   measure=costs.measure, probes=(probe,))
    costs.completed(execution.observations)
    actual = measured("verify_after", lambda: verified(execution.prepared))
    scored, = execution.scores
    if scored.truncated != selected.truncated:
        raise ValueError("Native scoring completed at a different source stopping boundary")
    result = observation(runtime, selected, scored=scored, actual=actual)
    values = {"binding": call.invocation.binding(),
              "observation": full_vocabulary(result, selected, observed=scored.probe, measurements=measurements, stream=True)}
    if probe is None:
        transcript.emit("score_result", values)
    else:
        transcript.emit_stream("score_result", values, encode=probeoutput.chunks)


def run(options, *, engine_factory=entry.factory, source=None, output=None):
    incoming = sys.stdin if source is None else source
    transcript = Transcript(sys.stdout if output is None else output)
    envelope = core.decode(incoming.readline())
    call, _, _ = decode(envelope)
    measurements = []

    def emit(stage, values):
        measurements.append({"stage": stage, **values})
        transcript.emit(stage, values)

    with redirect_stdout(sys.stderr), ExitStack() as stack:
        loader, _, _ = entry.components(options, stack, emit=emit, engine_factory=engine_factory)
        runtime = loader(options.cache, options.adapter, expected=call.identities)
        execute(runtime, envelope, incoming=incoming, transcript=transcript,
                measured=partial(measure, emit=emit), emit=emit, measurements=measurements)


def main():
    run(entry.batch_arguments(__doc__))


if __name__ == "__main__":
    main()
