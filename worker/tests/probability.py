import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import torch

from .cohort import request
from compare import claim
from learning import update
from .learning import batch, make_learner
from objective import Tokens
from probability import ROLES, capture, save, words
from reader import Input, compare, snapshot

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
    for label in numerical["order"]:
        behavior = torch.tensor(originals[label]["behavior_bits"], dtype=torch.int64).int().view(torch.float32)
        tokens = Tokens(behavior=behavior, proximal=behavior, reference=behavior, current=behavior,
                        advantage=torch.ones_like(behavior), active=torch.ones_like(behavior, dtype=torch.bool))
        observations.append(capture(label, tokens))
    path = directory / f"{name}.json"
    digest = save(path, observations, invocation=bound, request=numerical)
    consumed = {"stage": "consumed", **bound, "request": numerical}
    result = {"stage": "result", "binding": bound["binding"], "request": numerical,
              "gradients": "f" * 64, "probabilities": digest}
    log = directory / f"{name}.jsonl"
    log.write_text("\n".join(json.dumps(event) for event in (consumed, result)) + "\n")
    return Input(probabilities=path, log=log, call=call)


def rehash(observation, encoded):
    observation.probabilities.write_bytes(encoded)
    events = [json.loads(line) for line in observation.log.read_text().splitlines()]
    events[-1]["probabilities"] = hashlib.sha256(encoded).hexdigest()
    observation.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")


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

    def test_delivery_permutation_and_distinct_bindings_preserve_words(self):
        left = fixture(self.directory, "left", call=1)
        right = fixture(self.directory, "right", call=2, reverse=True)
        self.assertTrue(compare(left, right)["equal"])
        self.assertNotEqual(left.probabilities.read_bytes(), right.probabilities.read_bytes())

    def test_real_word_changes_are_reported_and_cli_exits_nonzero(self):
        left = fixture(self.directory, "left", call=1)
        right = fixture(self.directory, "right", call=2)
        value = json.loads(right.probabilities.read_bytes())
        value["samples"][0]["current"] = [NEGATIVE_ZERO]
        rehash(right, json.dumps(value).encode())
        result = compare(left, right)
        self.assertFalse(result["equal"])
        self.assertEqual(result["differences"], [{"sample": "a", "fields": ["current"]}])
        command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "reader.py")]
        for side, item in (("left", left), ("right", right)):
            command += [f"--{side}-probabilities", str(item.probabilities), f"--{side}-log", str(item.log),
                        f"--{side}-call", str(item.call)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
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
        for changed in (b'{"format":"invar-probabilities-v1",' + encoded[1:], encoded + b" null",
                        encoded.replace(b'"dtype": "F32"', b'"dtype":"F32","dtype":"F32"')):
            rehash(observation, changed)
            with self.assertRaises(ValueError):
                snapshot(observation.probabilities, claim(observation.log, observation.call))

    def test_changed_bytes_fail_before_parsing(self):
        observation = fixture(self.directory, "input", call=1)
        observation.probabilities.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "reported digest"):
            snapshot(observation.probabilities, claim(observation.log, observation.call))


if __name__ == "__main__":
    unittest.main()
