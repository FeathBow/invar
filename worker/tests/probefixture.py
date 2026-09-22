import hashlib
import json

from worker import scalar
from worker.distribution import FP32_BYTES, PROBE_FORMAT
from worker.probeschema import CLIENT_SCOPE, NATIVE_CACHE, NATIVE_PATH, VLLM_ENGINE, WORKER_SCOPE, native_relation


def artifact(*, steps=(0,), vocabulary=2, backend="vllm", prompt="fixture"):
    horizon = steps[-1] + 1
    word = scalar.word(1 / vocabulary)
    log_word = scalar.word(-1.)
    source = {"log_sha256": "a" * 64, "binding": {"call": 0, "attempt": 0, "instance": 0},
              "tokens": [1, *([0] * horizon)], "behavior_bits": [log_word] * horizon, "prompt_length": 1,
              "text": "fixture", "truncated": True, "model": "protocol-fixture", "revision": "v1",
              "adapter": "a" * 64, "tokenizer": "b" * 64, "base": "c" * 64, "assembly": "d" * 64,
              "request": {"prompt": prompt, "tokens": horizon, "temperature": 1., "seed": 0}}
    encoded = json.dumps(source, ensure_ascii=False)
    execution, measurements = backend_fields(backend, horizon)
    return {"format": PROBE_FORMAT, "role": "cached_behavior_full_vocabulary", "use_admission": "not_evaluated",
            "source_inspection": encoded, "source": source,
            "source_inspection_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "target": {**{key: source[key] for key in ("adapter", "tokenizer", "base", "assembly", "model", "revision")},
                       "numerics": "protocol-fixture"}, "request": source["request"], "prefix_tokens": [1],
            "response_tokens": [0] * horizon, "log_probability_bits": [log_word] * horizon,
            "probability": {"role": "behavior", "log_base": "e", "representation": "F32 words", "zero_support_word": 0xff800000,
                            "temperature": 1., "mask": "none", "top_k": "disabled", "top_p": "disabled"},
            "execution": execution,
            "implementation": {"sources_sha256": {"protocol-fixture": "e" * 64}, "packages": {"fixture": "v1"}},
            "full_vocabulary": {"steps": list(steps), "vocabulary": vocabulary, "coordinates": "output token ids 0..vocabulary-1",
                                "representation": "F32 probability words",
                                "snapshots": [{"step": step, "probability_bits": [word] * vocabulary} for step in steps],
                                "raw_payload_bytes": len(steps) * vocabulary * FP32_BYTES},
            "measurements": measurements}


def backend_fields(backend, horizon):
    if backend == "vllm":
        execution = {"engine": VLLM_ENGINE, "cache_origin": NATIVE_CACHE, "path_control": NATIVE_PATH,
                     "native_sample_rows": horizon + 1, "ignored_prefill_rows": 1, "truncated": True,
                     "distribution": native_relation()}
        measured = {"stage": "cross_score", "seconds": 0.1, "seconds_scope": CLIENT_SCOPE, "allocator": "torch.cuda",
                    "workers": [{"host": "protocol-fixture", "pid": 1, "device": "cuda:0", "seconds": 0.1,
                                 "peak_allocated": 64, "peak_reserved": 128, "scope": WORKER_SCOPE}]}
    elif backend == "mlx":
        execution = {"engine": "mlx_lm.generate.BatchGenerator", "cache_origin": "fresh native caches; no reference cache input",
                     "unused_native_lookahead_draws": 1, "truncated": True, "sampling": {"batch_size": 1, "prefill_step": 2}}
        measured = {"stage": "cross_score", "seconds": 0.1, "allocator": "mlx", "peak_active": 64, "cache_end": 0}
    else:
        raise ValueError("Unknown synthetic backend fixture")
    return execution, [measured]
