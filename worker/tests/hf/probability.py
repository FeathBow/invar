import os
import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import torch

from worker.tests.hf.cohort import request
from worker.compare import claim
from worker.hf.learning import update
from worker.tests.hf.learning import batch, make_learner
from worker.hf.objective import Tokens
from worker.hf.probability import ROLES, capture, checked, cotangents, loss, save, words
from worker.reader import Input, compare, snapshot
from worker import scalar

NEGATIVE_ZERO = 0x80000000
NAN_WORD = 0x7FC00000
POSITIVE_ONE = 0x3F800000


def fixture(directory, name, *, call, reverse=False):
    numerical = request()
    if reverse:
        numerical["samples"].reverse()
    bound = {"binding": {"call": call, "attempt": call, "instance": call}, "program": "fixed program"}
    originals = {item["sample"]: item for item in numerical["samples"]}
    observations = []
    profile = scalar.Profile(epsilon=numerical["epsilon"], penalty=numerical["penalty"])
    count = sum(len(item["behavior_bits"]) for item in originals.values())
    for label in numerical["order"]:
        behavior = torch.tensor(originals[label]["behavior_bits"], dtype=torch.int64).int().view(torch.float32)
        tokens = Tokens(behavior=behavior, proximal=behavior, reference=behavior, current=behavior,
                        advantage=torch.full_like(behavior, scalar.number(originals[label]["advantage_bits"])),
                        active=torch.ones_like(behavior, dtype=torch.bool))
        evaluated, _, _ = cotangents(capture(label, tokens), profile, total=count, device=behavior.device)
        observations.append(evaluated)
    path = directory / f"{name}.json"
    digest = save(path, observations, invocation=bound, request=numerical)
    consumed = {"stage": "consumed", **bound, "request": numerical}
    result = {"stage": "result", "binding": bound["binding"], "request": numerical,
              "gradients": "f" * 64, "probabilities": digest,
              "update": {"loss": scalar.number(loss(observations)), "active_tokens": count}}
    log = directory / f"{name}.jsonl"
    log.write_text("\n".join(json.dumps(event) for event in (consumed, result)) + "\n")
    return Input(probabilities=path, log=log, call=call)


def rehash(observation, encoded):
    observation.probabilities.write_bytes(encoded)
    events = [json.loads(line) for line in observation.log.read_text().splitlines()]
    events[-1]["probabilities"] = hashlib.sha256(encoded).hexdigest()
    observation.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def recompute(value):
    updated = copy.deepcopy(value)
    profile = scalar.Profile(epsilon=updated["request"]["epsilon"], penalty=updated["request"]["penalty"])
    count = sum(len(item["active"]) for item in updated["samples"])
    for item in updated["samples"]:
        inputs = tuple(scalar.Inputs(**dict(zip(ROLES, values, strict=True)))
                       for values in zip(*(item[role] for role in ROLES), strict=True))
        item["objective"] = scalar.document(scalar.calculate(profile, count, inputs))
    updated["loss"] = scalar.mean32(tuple(term for item in updated["samples"] for term in item["objective"]["terms"]))
    return updated


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-probability-"))

    def test_actual_update_records_the_pre_step_values_in_logical_order(self):
        logical = batch()
        logical = replace(logical, samples=tuple(replace(item, advantage=1) for item in logical.samples))
        learner = make_learner()
        before = tuple(words(learner.evaluate(learner.model, item.trajectory)) for item in logical.samples)
        result = update(learner, logical)
        self.assertEqual(tuple(item.sample for item in result.probabilities), logical.order)
        self.assertEqual(tuple(item.words[ROLES.index("current")] for item in result.probabilities), before)
        after = tuple(words(learner.evaluate(learner.model, item.trajectory)) for item in logical.samples)
        self.assertNotEqual(before, after)
        for delivery in permutations(logical.samples):
            received = update(make_learner(), replace(logical, samples=delivery))
            self.assertEqual(received.probabilities, result.probabilities)

    def test_capture_preserves_signed_zero_and_does_not_alias_tensors(self):
        for dtype, expected in ((torch.float32, NEGATIVE_ZERO), (torch.float64, 1 << 63)):
            with self.subTest(dtype=dtype):
                value = torch.tensor([-0.0], dtype=dtype)
                tokens = Tokens(**dict.fromkeys(ROLES, value), active=torch.tensor([True]))
                observed = capture("slash/name", tokens)
                value.fill_(1)
                self.assertEqual(observed.words, ((expected,),) * len(ROLES))
                self.assertEqual(observed.active, (True,))

    def test_actual_objective_check_requires_words_dtype_and_complete_response_mask(self):
        value = torch.tensor([-0.0, -0.0], dtype=torch.float32)
        tokens = Tokens(**dict.fromkeys(ROLES, value), active=torch.tensor([True, True]))
        observed = checked("a", tokens, advantage=-0.0, count=2)
        self.assertEqual(observed.words[ROLES.index("advantage")], (NEGATIVE_ZERO,) * 2)
        changes = [replace(tokens, advantage=torch.zeros_like(value)),
                   replace(tokens, advantage=value.double()),
                   replace(tokens, current=value[:1]),
                   replace(tokens, active=torch.tensor([True, False])),
                   replace(tokens, active=torch.tensor([1, 1]))]
        for changed in changes:
            with self.subTest(tokens=changed), self.assertRaises(ValueError):
                checked("a", changed, advantage=-0.0, count=2)
        for count in (0, 1, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                checked("a", tokens, advantage=-0.0, count=count)

    def test_delivery_permutation_and_distinct_bindings_preserve_words(self):
        left = fixture(self.directory, "left", call=1)
        right = fixture(self.directory, "right", call=2, reverse=True)
        self.assertTrue(compare(left, right)["equal"])
        self.assertNotEqual(left.probabilities.read_bytes(), right.probabilities.read_bytes())

    def test_selected_core_is_required_without_a_fallback(self):
        executable = shutil.which("invar")
        self.assertIsNotNone(executable, "Build invar and add it to this test process's PATH")
        left = fixture(self.directory, "selected-left", call=1)
        right = fixture(self.directory, "selected-right", call=2)
        self.assertEqual(compare(left, right, core_executable=executable), compare(left, right))
        with self.assertRaises(FileNotFoundError):
            compare(left, right, core_executable=self.directory / "missing-invar")

    def test_a_claim_binds_the_complete_log_snapshot(self):
        observation = fixture(self.directory, "snapshot", call=1)
        expected = claim(observation.log, observation.call)
        before = snapshot(observation.probabilities, expected)
        with observation.log.open("a") as target:
            target.write('{"stage":"profile"}\n')
        with self.assertRaisesRegex(ValueError, "log identity changed after inspection"):
            snapshot(observation.probabilities, expected)
        self.assertEqual(snapshot(observation.probabilities, claim(observation.log, observation.call)), before)

    def test_reported_active_token_count_is_checked_by_the_core(self):
        observation = fixture(self.directory, "count", call=1)
        events = [json.loads(line) for line in observation.log.read_text().splitlines()]
        events[-1]["update"]["active_tokens"] += 1
        observation.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        with self.assertRaisesRegex(ValueError, "scalar observation token count"):
            snapshot(observation.probabilities, claim(observation.log, observation.call))

    def test_summary_float_zero_sign_is_not_lost_in_json_parsing(self):
        observation = fixture(self.directory, "loss-sign", call=1)
        events = [json.loads(line) for line in observation.log.read_text().splitlines()]
        self.assertEqual(events[-1]["update"]["loss"], 0.0)
        events[-1]["update"]["loss"] = -0.0
        observation.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        with self.assertRaisesRegex(ValueError, "summary differs"):
            snapshot(observation.probabilities, claim(observation.log, observation.call))
        events[-1]["update"]["loss"] = 0
        observation.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        accepted = snapshot(observation.probabilities, claim(observation.log, observation.call))
        encoded = observation.log.read_bytes().replace(b'"update":', b'"upd\\u0061te":')
        encoded = encoded.replace(b'"loss": 0', b'"l\\u006fss": -0')
        observation.log.write_bytes(encoded)
        self.assertEqual(snapshot(observation.probabilities, claim(observation.log, observation.call)), accepted)

    def test_real_word_changes_are_reported_and_cli_exits_nonzero(self):
        left = fixture(self.directory, "left", call=1)
        right = fixture(self.directory, "right", call=2)
        value = json.loads(right.probabilities.read_bytes())
        value["samples"][0]["current"] = [NEGATIVE_ZERO]
        value = recompute(value)
        rehash(right, json.dumps(value).encode())
        events = [json.loads(line) for line in right.log.read_text().splitlines()]
        events[-1]["update"]["loss"] = scalar.number(value["loss"])
        right.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        result = compare(left, right)
        self.assertFalse(result["equal"])
        self.assertEqual(result["differences"], [{"sample": "a", "fields": ["current", "objective"]}])
        command = [sys.executable, "-B", "-m", "worker.reader"]
        for side, item in (("left", left), ("right", right)):
            command += [f"--{side}-probabilities", str(item.probabilities), f"--{side}-log", str(item.log),
                        f"--{side}-call", str(item.call)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False, cwd=Path(__file__).resolve().parents[3])
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(json.loads(process.stdout), result)

    def test_matching_hash_cannot_hide_invalid_roles_masks_or_words(self):
        observation = fixture(self.directory, "input", call=1)
        original = json.loads(observation.probabilities.read_bytes())
        changes = [("dtype", "F64"), ("sample", "unknown"), ("behavior", [0]), ("active", [False]),
                   ("active", [1]), ("current", []), ("current", [True]), ("current", [-1]),
                   ("current", [1 << 32]), ("current", [NAN_WORD]), ("current", [POSITIVE_ONE]),
                   ("advantage", [NAN_WORD])]
        for key, value in changes:
            changed = copy.deepcopy(original)
            changed["samples"][0][key] = value
            with self.subTest(key=key, value=value):
                self.reject(observation, changed)
        for role in ROLES:
            changed = copy.deepcopy(original)
            del changed["samples"][0][role]
            self.reject(observation, changed)

    def reject(self, observation, value):
        rehash(observation, json.dumps(value).encode())
        with self.assertRaises(ValueError):
            snapshot(observation.probabilities, claim(observation.log, observation.call))

    def test_bound_metadata_sample_inventory_and_strict_json_are_mandatory(self):
        observation = fixture(self.directory, "input", call=1)
        original = json.loads(observation.probabilities.read_bytes())
        for key in original:
            missing = copy.deepcopy(original)
            del missing[key]
            self.reject(observation, missing)
        variants = [{**original, "request": {}}, {**original, "invocation": {}},
                    {**original, "samples": []}, {**original, "samples": original["samples"][::-1]},
                    {**original, "samples": original["samples"] * 2}, {**original, "extra": 0}]
        for changed in variants:
            self.reject(observation, changed)
        encoded = json.dumps(original).encode()
        for changed in (b'{"format":"invar-probabilities-v2",' + encoded[1:], encoded + b" null",
                        encoded.replace(b'"dtype": "F32"', b'"dtype":"F32","dtype":"F32"')):
            rehash(observation, changed)
            with self.assertRaises(ValueError):
                snapshot(observation.probabilities, claim(observation.log, observation.call))

    def test_changed_bytes_fail_before_parsing(self):
        observation = fixture(self.directory, "input", call=1)
        observation.probabilities.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "reported digest"):
            snapshot(observation.probabilities, claim(observation.log, observation.call))

    def test_rehashed_scalar_terms_gradients_and_loss_are_independently_checked(self):
        observation = fixture(self.directory, "scalars", call=1)
        original = json.loads(observation.probabilities.read_bytes())
        for field, _ in scalar.FIELDS:
            changed = copy.deepcopy(original)
            changed["samples"][0]["objective"][field][0] ^= 1
            with self.subTest(field=field):
                self.reject(observation, changed)
        for field, value in (("scalar_reference", "unqualified"), ("loss", 1),
                             ("format", "invar-probabilities-v1")):
            self.reject(observation, {**original, field: value})
        rehash(observation, json.dumps(original).encode())
        events = [json.loads(line) for line in observation.log.read_text().splitlines()]
        events[-1]["update"]["loss"] = 0.5
        observation.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        with self.assertRaisesRegex(ValueError, "summary differs"):
            snapshot(observation.probabilities, claim(observation.log, observation.call))


if __name__ == "__main__":
    unittest.main()
