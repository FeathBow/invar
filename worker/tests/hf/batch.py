import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from worker.batch import FORMAT, approve, capture, decode, serve
from worker.tests.hf.inference import envelope


def requests():
    identities = {name: character * 64 for name, character in zip(
        ("adapter", "tokenizer", "base", "assembly"), "abcd", strict=True)}
    return [envelope(identities, index) for index in (2, 0, 1)]


def frame(calls):
    return {"format": FORMAT, "adapter": "an adapter path", "calls": [json.dumps(call) for call in calls]}


def permissions(calls):
    return {"format": FORMAT, "permissions": [json.dumps({key: call[key] for key in ("binding", "program")}) for call in calls]}


class BatchTests(unittest.TestCase):
    def test_original_calls_and_exact_program_text_survive_framing(self):
        values = requests()
        values[1]["program"] += "\n\\\"Unicode: 中文"
        actual = decode(json.dumps(frame(values)))
        self.assertEqual(actual.adapter, Path("an adapter path"))
        self.assertEqual([call.invocation.call for call in actual.calls], [2, 0, 1])
        self.assertEqual(actual.calls[1].invocation.program, values[1]["program"])
        approve(tuple(call.invocation for call in actual.calls), source=io.StringIO(json.dumps(permissions(values)) + "\n"))

    def test_a_bad_later_member_is_rejected_before_model_loading(self):
        def forbidden(*args, **kwargs):
            self.fail("Malformed finite input reached the numerical boundary")

        for field in ("call", "attempt", "instance"):
            values = requests()
            values[-1]["binding"][field] = values[0]["binding"][field]
            values[-1]["load"]["binding"][field] = values[0]["binding"][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reuses"):
                serve(SimpleNamespace(cache=Path("unused")), source=io.StringIO(json.dumps(frame(values)) + "\n"),
                      loader=forbidden, execute=forbidden, permission=forbidden)
        values = requests()
        values[-1]["request"]["temperature"] = 0
        with self.assertRaisesRegex(ValueError, "positive temperature"):
            serve(SimpleNamespace(cache=Path("unused")), source=io.StringIO(json.dumps(frame(values)) + "\n"),
                  loader=forbidden, execute=forbidden, permission=forbidden)

    def test_finite_input_requires_strict_nonempty_encoded_call_inventory(self):
        original = frame(requests())
        invalid = ({**original, "calls": []}, {**original, "calls": requests()},
                   {**original, "calls": ["null"]}, {**original, "adapter": ""},
                   {**original, "format": "other"}, {**original, "extra": True})
        for changed in invalid:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                decode(json.dumps(changed))
        duplicate = json.dumps(original).replace('"format":', '"format":"duplicate","format":', 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            decode(duplicate)
        repeated = requests()
        encoded = json.dumps(repeated[-1]).replace('"tokens":', '"tokens":1,"tokens":')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            decode(json.dumps({**original, "calls": [*original["calls"][:-1], encoded]}))

    def test_permission_inventory_is_complete_distinct_ordered_and_exact(self):
        values = requests()
        expected = tuple(call.invocation for call in decode(json.dumps(frame(values))).calls)
        original = permissions(values)
        members = original["permissions"]
        wrong = {**values[-1], "program": "wrong"}
        invalid = [[], members[:-1], members + members[:1], members[::-1],
                   members[:1] * 2 + members[2:], members[:-1] + permissions([wrong])["permissions"]]
        for changed in invalid:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "inventory"):
                approve(expected, source=io.StringIO(json.dumps({**original, "permissions": changed})))
        for changed in ({**original, "format": "other"}, {**original, "permissions": values}, {**original, "extra": True}):
            with self.assertRaises(ValueError):
                approve(expected, source=io.StringIO(json.dumps(changed)))
        duplicated = members[-1].replace('"program":', '"program":"wrong","program":')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            approve(expected, source=io.StringIO(json.dumps({**original, "permissions": members[:-1] + [duplicated]})))

    def test_observation_capture_keeps_original_floating_literals(self):
        output = capture(lambda emit: emit("result", {"behavior": [-0.0], "text": "中文"}))
        self.assertIn('[-0.0]', output)
        self.assertTrue(output.endswith("\n"))
        self.assertEqual(json.loads(output)["text"], "中文")
