import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from dataclasses import asdict, replace
from functools import partial
import gc
import hashlib
import io
import json
import unittest
import weakref

import torch

from worker.learner import FORMAT, serve
from worker.resident import FORMAT as RESIDENT_FORMAT, Owner, Transcript
from worker.hf.probability import logprobs
from worker.tests.hf.inference import IDENTITY, model
from worker.tests.hf.learner import fixture
from worker.tests.hf.handshake import cpu_measure
from worker.tests.responder import request_core
from worker.tests.hf.tokenization import make_tokenizer


def invocation(value):
    return {"binding": value.binding(), "program": value.program}


class Peer:
    def __init__(self, call, paths, output, *, groups=2, invalid=None):
        self.call, self.paths, self.output = call, paths, output
        self.groups, self.invalid = groups, invalid
        self.position, self.start = 0, 0
        self.owner = Owner(role="learning", session=0)
        self.messages = []
        self.core, self.seen = None, 0

    def readline(self):
        if self.core is not None:
            lines = self.output.getvalue().splitlines()
            answers = [answer for line in lines[self.seen:] if (answer := self.core.reply(line))]
            self.seen = len(lines)
            if answers:
                return answers[-1]
        position = self.position
        self.position += 1
        group, phase = divmod(position, 3)
        if group >= self.groups:
            value = {"format": RESIDENT_FORMAT, "owner": self.owner.value(), "action": "close"}
            return json.dumps(value) + "\n" if position == 3 * self.groups else ""
        if phase == 0:
            value = self.request(group)
        elif phase == 1:
            value = invocation(self.call.invocation)
        else:
            raw = self.output.getvalue()[self.start:]
            value = {"format": RESIDENT_FORMAT, "owner": self.owner.value(), "action": "release",
                     "loads": [invocation(self.call.load)], "result_sha256": hashlib.sha256(raw.encode()).hexdigest()}
            if self.invalid == "release":
                value["result_sha256"] = "0" * 64
        self.messages.append(value)
        return json.dumps(value) + "\n"

    def request(self, group):
        checkpoint = self.paths.checkpoint
        output = self.paths.output.parent / ("service" + str(group))
        if group:
            result = [value for line in self.output.getvalue().splitlines() if (value := json.loads(line))["stage"] == "result"][-1]
            ordinal = {name: getattr(self.call.invocation, name) + 100 for name in ("call", "attempt", "instance")}
            if self.invalid == "replay":
                ordinal["instance"] = self.call.invocation.instance
            self.call = replace(self.call, invocation=replace(self.call.invocation, **ordinal),
                                load=replace(self.call.load, **ordinal),
                                request=replace(self.call.request, policy=result["adapter"], learner=result["learner"],
                                                samples=tuple(replace(item, reference_bits=item.behavior_bits)
                                                              for item in self.call.request.samples)))
            checkpoint = self.paths.output.parent / ("service" + str(group - 1))
        self.start = len(self.output.getvalue())
        self.core, self.seen = request_core(self.call.request), len(self.output.getvalue().splitlines())
        call = {"invocation": invocation(self.call.invocation), "load": invocation(self.call.load),
                "request": asdict(self.call.request)}
        return {"format": FORMAT, "checkpoint": str(checkpoint), "output": str(output), "call": json.dumps(call)}


def execute(*, groups=2, invalid=None):
    call, paths = fixture()
    output = io.StringIO()
    transcript = Transcript(output)
    source = Peer(call, paths, output, groups=groups, invalid=invalid)
    loaded, closed = [], []

    def loader(options, request):
        actual = cpu_measure("load", model, emit=transcript.emit)
        loaded.append(weakref.ref(actual))
        return actual, make_tokenizer(), IDENTITY

    def close():
        gc.collect()
        closed.append(all(reference() is None for reference in loaded))

    def run():
        serve(source.owner, paths, source=source, transcript=transcript, loader=loader,
              measure=cpu_measure, evaluate=partial(logprobs, device="cpu"), close=close)

    return run, source, paths, loaded, closed


class LearnerServiceTests(unittest.TestCase):
    def test_actual_updates_release_and_close_one_loaded_model(self):
        run, source, paths, loaded, closed = execute()
        run()
        records = [json.loads(line) for line in source.output.getvalue().splitlines()]
        stages = [value["stage"] for value in records]
        self.assertEqual(stages.count("load"), 1)
        self.assertEqual(stages.count("activation"), 2)
        self.assertEqual(stages.count("result"), 2)
        self.assertEqual(stages.count("released"), 2)
        self.assertEqual(stages[-1], "closed")
        self.assertEqual((len(loaded), closed), (1, [True]))
        self.assertEqual(records[-1]["groups"], 2)
        for value in records:
            if value["stage"] in ("released", "closed"):
                measured = json.loads(value["measurement"])
                self.assertEqual(measured["stage"], value["stage"])
                self.assertGreaterEqual(measured["cpu_seconds"], 0)
        original = [value for value in source.messages if value.get("action") == "release"]
        acknowledged = [value for value in records if value["stage"] == "released"]
        self.assertEqual([value["result_sha256"] for value in original], [value["result_sha256"] for value in acknowledged])
        self.assertEqual([value["loads"] for value in original], [value["loads"] for value in acknowledged])
        saved = torch.load(paths.output.parent / "service1/learner.pt", weights_only=True)
        self.assertTrue(all(state["step"].item() == 2 for state in saved["optimizer"]["state"].values()))

    def test_invalid_release_has_no_acknowledgement_or_second_update(self):
        run, source, paths, _, closed = execute(invalid="release")
        with self.assertRaisesRegex(ValueError, "release differs"):
            run()
        stages = [json.loads(line)["stage"] for line in source.output.getvalue().splitlines()]
        self.assertEqual(stages.count("result"), 1)
        self.assertNotIn("released", stages)
        self.assertFalse((paths.output.parent / "service1").exists())
        self.assertEqual(closed, [])

    def test_replayed_activation_stops_before_second_output_or_consumption(self):
        run, source, paths, loaded, _ = execute(invalid="replay")
        with self.assertRaisesRegex(ValueError, "historical"):
            run()
        stages = [json.loads(line)["stage"] for line in source.output.getvalue().splitlines()]
        self.assertEqual((stages.count("consumed"), stages.count("released"), len(loaded)), (1, 1, 1))
        self.assertFalse((paths.output.parent / "service1").exists())

    def test_unused_owner_closes_without_loading_a_model(self):
        run, source, _, loaded, closed = execute(groups=0)
        run()
        records = [json.loads(line) for line in source.output.getvalue().splitlines()]
        self.assertEqual([value["stage"] for value in records], ["closed"])
        self.assertEqual(records[0]["groups"], 0)
        self.assertEqual((loaded, closed), ([], [True]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
