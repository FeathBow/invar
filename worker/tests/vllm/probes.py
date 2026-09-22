from dataclasses import replace
import io
import pickle
import struct
from types import SimpleNamespace
import unittest

import torch

from worker.distribution import Observed, Probe, Snapshot
from worker.probeoutput import chunks
from worker.probepacked import Packed, Snapshots, packed
from worker.resident import Transcript
from worker.scoring import TokenPath
from worker.tests.vllm.distribution import competition, words
from worker.tests.vllm.prescribed import binding, observation
from worker.vllm.scoreobservation import observed, probe_selection, trace_score
from worker.vllm.prescribed import Prescribed, sampled
from worker.vllm.scoring import full_vocabulary


def sample(control, logits, *, requests, active, history):
    monitor = observation(*requests, active=active, history=history)
    monitor.permitted = monitor.logits_finished = True
    def operation(actual, metadata):
        tokens, logs, _ = competition(actual, torch.Generator().manual_seed(73))
        return tokens, logs
    selected, logs = sampled(operation, logits, SimpleNamespace(logprob_token_ids={}),
                             control=control, monitor=monitor)
    control.observe(selected.to(torch.int32).unsqueeze(-1), logs.gather(-1, selected.unsqueeze(-1)))
    for key, token in zip(requests, selected.tolist(), strict=True):
        if key in active:
            history[key].append(token)
    return logs


class OwnedProbeTests(unittest.TestCase):
    def test_response_steps_follow_owned_rows_across_prefill_and_reordering(self):
        control = Prescribed({"a": binding(2), "b": binding(2)}, {"a": [1, 0], "b": [0, 3]},
                             probes={"a": Probe(steps=(0, 1)), "b": Probe(steps=(0,))})
        history = {"a": [], "b": []}
        first = torch.tensor([[1., 2., 3., 4.], [-120., 0., -1., -2.]])
        second = torch.tensor([[2., 0., 3., 1.], [4., 2., -1., 0.]])
        third = torch.tensor([[1., 0., -2., 3.]])
        logs = sample(control, first, requests=("a", "b"), active=("b",), history=history)
        self.assertTrue(torch.isfinite(logs[1, 0]))
        sample(control, second, requests=("b", "a"), active=("a", "b"), history=history)
        sample(control, third, requests=("a",), active=("a",), history=history)
        control.completed(history)
        a, b = (control.accounting(key, completed=True) for key in ("a", "b"))
        self.assertEqual((a["native_sample_rows"], a["ignored_prefill_rows"]), (3, 1))
        self.assertEqual((b["native_sample_rows"], b["ignored_prefill_rows"]), (2, 0))
        self.assertEqual(tuple(item.probability_bits for item in a["probe"].snapshots),
                         (words(second.softmax(-1)[1]), words(third.softmax(-1)[0])))
        self.assertEqual(b["probe"].snapshots[0].probability_bits, words(first.softmax(-1)[1]))
        self.assertEqual(b["probe"].snapshots[0].probability_bits[0], 0)

    def test_undeclared_request_and_unselected_step_require_no_capture(self):
        control = Prescribed({"a": binding(2), "b": binding(1)}, {"a": [1, 0], "b": [0]},
                             probes={"a": Probe(steps=(1,))})
        history = {"a": [], "b": []}
        logits = torch.tensor([[1., 2., 0., 3.]])
        for key in ("b", "a"):
            monitor = observation(key, active=(key,), history=history)
            sentinel = object()
            result = control.probes.sample(lambda: sentinel, logits=logits, monitor=monitor)
            self.assertIs(result, sentinel)
            sample(control, logits, requests=(key,), active=(key,), history=history)
        sample(control, logits, requests=("a",), active=("a",), history=history)
        control.completed(history)
        self.assertNotIn("probe", control.accounting("b", completed=True))
        self.assertEqual(control.accounting("a", completed=True)["probe"].probe.steps, (1,))

    def test_denied_permission_never_executes_or_captures(self):
        control = Prescribed({"a": binding(1)}, {"a": [0]}, probes={"a": Probe(steps=(0,))})
        monitor = observation("a", active=("a",), history={"a": []})
        monitor.permitted = False
        monitor.logits_finished = True
        calls = []
        with self.assertRaisesRegex(RuntimeError, "no permitted"):
            sampled(lambda *args: calls.append(True), torch.zeros((1, 4)),
                    SimpleNamespace(logprob_token_ids={}), control=control, monitor=monitor)
        self.assertEqual(calls, [])
        self.assertEqual(control.probes.snapshots, {"a": []})

    def test_sampler_failure_preserves_original_error_and_abort_has_no_partial_probe(self):
        control = Prescribed({"a": binding(1)}, {"a": [0]}, probes={"a": Probe(steps=(0,))})
        monitor = observation("a", active=("a",), history={"a": []})
        monitor.permitted = monitor.logits_finished = True
        original = ValueError("failed after actual mass computation")
        def operation(logits, metadata):
            logits.softmax(-1)
            raise original
        with self.assertRaises(ValueError) as caught:
            sampled(operation, torch.zeros((1, 4)), SimpleNamespace(logprob_token_ids={}),
                    control=control, monitor=monitor)
        self.assertIs(caught.exception, original)
        self.assertEqual(control.probes.snapshots, {"a": []})
        self.assertNotIn("probe", control.accounting("a", completed=False))
        sample(control, torch.zeros((1, 4)), requests=("a",), active=("a",), history=monitor.tokens)
        control.completed(monitor.tokens)
        self.assertEqual(len(control.accounting("a", completed=True)["probe"].snapshots), 1)

    def test_missing_or_repeated_selected_position_cannot_complete(self):
        for repetitions in (0, 2):
            control = Prescribed({"a": binding(1)}, {"a": [0]}, probes={"a": Probe(steps=(0,))})
            monitor = observation("a", active=("a",), history={"a": []})
            logits = torch.zeros((1, 4))
            for _ in range(repetitions):
                control.probes.sample(lambda: logits.softmax(-1), logits=logits, monitor=monitor)
            with self.subTest(repetitions=repetitions), self.assertRaisesRegex(ValueError, "exactly"):
                control.probes.completed()
            self.assertIsNone(control.probes.finished)

    def test_invalid_probe_scope_is_rejected(self):
        for declared in ({}, {"b": Probe(steps=(0,))}, {"a": (0,)}, {"a": Probe(steps=(1,))}):
            with self.subTest(declared=declared), self.assertRaises(ValueError):
                Prescribed({"a": binding(1)}, {"a": [0]}, probes=declared)


class ProbeDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.path = TokenPath(prefix=(1,), response=(0,))
        self.probe = Probe(steps=(0,))
        self.vector = Observed(probe=self.probe, snapshots=(Snapshot(step=0, probability_bits=(0, 0x3f800000)),))
        self.trace = {"tokens": [0], "behavior": [-120.], "native_sample_rows": 2, "ignored_prefill_rows": 1,
                      "probe": self.vector}
        self.delivered = (words(torch.tensor([-120.])), True)

    def test_finite_reported_log_and_zero_mass_remain_distinct_observations(self):
        result = trace_score(self.trace, self.path, self.delivered, probe=self.probe)
        self.assertEqual(result.probe, self.vector)
        self.assertEqual(result.log_probability_bits, self.delivered[0])

    def test_undeclared_missing_or_mismatched_observation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "undeclared"):
            trace_score(self.trace, self.path, self.delivered)
        mismatched = Observed(probe=Probe(steps=(1,)), snapshots=(replace(self.vector.snapshots[0], step=1),))
        for value in (None, {"probe": self.probe}, mismatched):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "declared response steps"):
                trace_score({**self.trace, "probe": value}, self.path, self.delivered, probe=self.probe)

    def test_distinct_worker_vectors_cannot_form_one_score(self):
        queued = (SimpleNamespace(internal="a"),)
        first = {"requests": {"a": self.trace}, "steps": [1]}
        different = Observed(probe=self.probe, snapshots=(Snapshot(step=0, probability_bits=(0x3f000000, 0x3f000000)),))
        second = {"requests": {"a": {**self.trace, "probe": different}}, "steps": [1]}
        self.assertEqual(len(observed((self.delivered,), queued, paths=(self.path,),
                                      observations=(first, first), probes=(self.probe,))), 1)
        with self.assertRaisesRegex(ValueError, "different prescribed-path"):
            observed((self.delivered,), queued, paths=(self.path,), observations=(first, second), probes=(self.probe,))

    def test_probe_declarations_are_checked_against_complete_path(self):
        self.assertEqual(probe_selection((self.path,), None), (None,))
        self.assertEqual(probe_selection((self.path,), (self.probe,)), (self.probe,))
        for supplied in ((), (self.probe, None), (object(),), (Probe(steps=(1,)),)):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                probe_selection((self.path,), supplied)

    def test_received_packed_words_are_checked_before_score_delivery(self):
        packet = Packed(probe=self.probe, snapshots=Snapshots(rows=tuple(map(packed, self.vector.snapshots))))
        result = trace_score({**self.trace, "probe": packet}, self.path, self.delivered, probe=self.probe)
        self.assertEqual(tuple(result.probe.snapshots), self.vector.snapshots)
        object.__setattr__(packet.snapshots.rows[0], "payload", struct.pack("<II", 0, 0x7fc00000))
        received = pickle.loads(pickle.dumps(packet, protocol=5))
        with self.assertRaises(ValueError):
            trace_score({**self.trace, "probe": received}, self.path, self.delivered, probe=self.probe)

    def test_packed_worker_agreement_compares_the_full_payload(self):
        queued = (SimpleNamespace(internal="a"),)
        packet = Packed(probe=self.probe, snapshots=Snapshots(rows=tuple(map(packed, self.vector.snapshots))))
        first = {"requests": {"a": {**self.trace, "probe": packet}}, "steps": [1]}
        changed = Packed(probe=self.probe, snapshots=Snapshots(rows=(packed(Snapshot(step=0, probability_bits=(0x3f000000, 0x3f000000))),)))
        second = {"requests": {"a": {**self.trace, "probe": changed}}, "steps": [1]}
        with self.assertRaisesRegex(ValueError, "different prescribed-path"):
            observed((self.delivered,), queued, paths=(self.path,), observations=(first, second), probes=(self.probe,))

    def test_streamed_transcript_matches_materialized_probe_bytes(self):
        packet = Packed(probe=self.probe, snapshots=Snapshots(rows=tuple(map(packed, self.vector.snapshots))))
        source = SimpleNamespace(inspection=b'{"source":"actual-fixture"}')
        before, after = io.StringIO(), io.StringIO()
        plain, streamed = Transcript(before), Transcript(after)
        result = {"execution": {"original": True}}
        plain.emit("score_result", full_vocabulary(result, source, observed=self.vector, measurements=[]))
        streamed.emit_stream("score_result", full_vocabulary(result, source, observed=packet, measurements=[], stream=True), encode=chunks)
        self.assertEqual(after.getvalue(), before.getvalue())
        self.assertEqual(streamed.digest.digest(), plain.digest.digest())


if __name__ == "__main__":
    unittest.main()
