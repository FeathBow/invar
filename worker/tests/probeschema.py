import hashlib
import json
import unittest

import torch

from worker import probe
from worker.probeschema import CLIENT_SCOPE, WORKER_SCOPE, NATIVE_CACHE, NATIVE_PATH, VLLM_ENGINE, native_relation
from worker.tests.vllm.distribution import words
from worker.vllm.resources import Scoring


def worker_cost():
    return {"host": "protocol-fixture", "pid": 1, "device": "cuda:0", "seconds": 0.1,
            "peak_allocated": 64, "peak_reserved": 128, "scope": WORKER_SCOPE}


def artifact():
    source = {"log_sha256": "a" * 64, "binding": {"call": 0, "attempt": 0, "instance": 0},
              "tokens": [1, 0], "behavior_bits": [0xbf800000], "prompt_length": 1,
              "text": "fixture", "truncated": True, "model": "protocol-fixture", "revision": "v1",
              "adapter": "a" * 64, "tokenizer": "b" * 64, "base": "c" * 64, "assembly": "d" * 64,
              "request": {"prompt": "fixture", "tokens": 1, "temperature": 1., "seed": 0}}
    encoded = json.dumps(source)
    logits = torch.tensor([[-120., 0.]])
    return {"format": "invar-cached-distribution-probe-v1", "role": "cached_behavior_full_vocabulary",
            "use_admission": "not_evaluated", "source_inspection": encoded, "source": source,
            "source_inspection_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "target": {**{key: source[key] for key in ("adapter", "tokenizer", "base", "assembly", "model", "revision")},
                       "numerics": "protocol-fixture"}, "request": source["request"], "prefix_tokens": [1], "response_tokens": [0],
            "log_probability_bits": [words(logits.log_softmax(-1))[0]],
            "probability": {"role": "behavior", "log_base": "e", "representation": "F32 words", "zero_support_word": 0xff800000,
                            "temperature": 1., "mask": "none", "top_k": "disabled", "top_p": "disabled"},
            "execution": {"engine": VLLM_ENGINE, "cache_origin": NATIVE_CACHE, "path_control": NATIVE_PATH,
                          "native_sample_rows": 2, "ignored_prefill_rows": 1, "truncated": True, "distribution": native_relation()},
            "implementation": {"sources_sha256": {"protocol-fixture": "e" * 64}, "packages": {"fixture": "v1"}},
            "full_vocabulary": {"steps": [0], "vocabulary": 2, "coordinates": "output token ids 0..vocabulary-1",
                                "representation": "F32 probability words", "snapshots": [{"step": 0, "probability_bits": list(words(logits.softmax(-1)))}],
                                "raw_payload_bytes": 8},
            "measurements": [{"stage": "cross_score", "seconds": 0.1, "seconds_scope": CLIENT_SCOPE,
                              "allocator": "torch.cuda", "workers": [worker_cost()]}]}


def read(value):
    return probe.read(json.dumps(value, allow_nan=False).encode())


class ProbeSchemaTests(unittest.TestCase):
    def test_underflow_is_preserved_and_kl_uses_the_mass_vector(self):
        target = read(artifact())
        self.assertEqual(target.vectors.snapshots[0].probability_bits, (0, 0x3f800000))
        reference = artifact()
        reference["full_vocabulary"]["snapshots"][0]["probability_bits"] = [0x3f000000, 0x3f000000]
        result = probe.compare(read(reference), target)
        self.assertEqual(result["observations"][0]["kl_reference_candidate"], {"kind": "positive_infinity"})
        self.assertEqual(result["candidate"]["workers"], [worker_cost()])
        self.assertNotIn("peak_active_bytes", result["candidate"])
        self.assertEqual(result["use_admission"], "not_evaluated")

    def test_capture_relation_is_explicit_and_cannot_be_relabeled_as_mlx(self):
        for field in ("mass_capture", "reported_log", "relation"):
            changed = artifact()
            changed["execution"]["distribution"][field] = "unsupported"
            with self.subTest(field=field), self.assertRaises(ValueError):
                read(changed)
        changed = artifact()
        changed["execution"].pop("distribution")
        with self.assertRaises(ValueError):
            read(changed)
        changed = artifact()
        changed["execution"] = {"engine": "mlx_lm.generate.BatchGenerator", "cache_origin": "fresh native caches; no reference cache input",
                                "unused_native_lookahead_draws": 1, "truncated": True, "sampling": {"batch_size": 1, "prefill_step": 2}}
        with self.assertRaisesRegex(ValueError, "zero support"):
            read(changed)

    def test_malformed_worker_scope_and_cost_cannot_become_measurements(self):
        invalid = (("host", ""), ("pid", True), ("pid", 0), ("device", ""), ("seconds", -1),
                   ("peak_allocated", 0), ("peak_reserved", 32), ("scope", "client"))
        for key, value in invalid:
            changed = artifact()
            changed["measurements"][0]["workers"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                read(changed)
        for workers in ([], [worker_cost(), worker_cost()]):
            changed = artifact()
            changed["measurements"][0]["workers"] = workers
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                read(changed)

    def test_shared_source_and_native_row_accounting_remain_required(self):
        for key, value in (("native_sample_rows", True), ("ignored_prefill_rows", -1), ("native_sample_rows", 3),
                           ("cache_origin", "source cache"), ("truncated", 1)):
            changed = artifact()
            changed["execution"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                read(changed)

    def test_client_timer_does_not_invoke_frontend_cuda_meter_for_probe(self):
        records, delegated = [], []
        def measured(stage, operation):
            delegated.append(stage)
            return operation()
        timing = Scoring(measured, emit=lambda stage, values: records.append({"stage": stage, **values}), probing=True)
        sentinel = object()
        self.assertIs(timing.measure("verify_before", lambda: sentinel), sentinel)
        self.assertIs(timing.measure("cross_score", lambda: sentinel), sentinel)
        self.assertEqual(records, [])
        self.assertEqual(delegated, ["verify_before"])
        timing.completed(({"probe_resources": worker_cost()},))
        self.assertEqual(records[0]["workers"], [worker_cost()])
        self.assertGreaterEqual(records[0]["seconds"], 0)

    def test_client_failure_never_emits_a_successful_cost_record(self):
        records = []
        timing = Scoring(None, emit=lambda *args: records.append(args), probing=True)
        error = ValueError("actual operation failure")
        def failed():
            raise error
        with self.assertRaises(ValueError) as caught:
            timing.measure("cross_score", failed)
        self.assertIs(caught.exception, error)
        with self.assertRaises(RuntimeError):
            timing.completed(({"probe_resources": worker_cost()},))
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
