import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .cohort import request
from compare import HEADER_BYTES, OBSERVATION, Input, Tensor, compare, complete, inventory
from probe import digest

PARAMETERS = ("base_model.model.0.lora_A", "base_model.model.0.lora_B")
GRADIENTS = tuple(f"{role}/{name}.default.weight" for role in ("objective", "reward") for name in PARAMETERS)


def encode(metadata, tensors):
    header = {"__metadata__": metadata}
    data = bytearray()
    for name, (dtype, shape, content) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(data), len(data) + len(content)]}
        data.extend(content)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % HEADER_BYTES)
    return len(encoded).to_bytes(HEADER_BYTES, "little") + encoded + data


def fixture(directory, name, *, call, tensor=None, change=None):
    numerical = request()
    numerical["policy"] = digest(load_file(directory / "policy.safetensors"))
    bound = {"call": call, "attempt": call, "instance": call}
    metadata = {"binding": json.dumps(bound), "program": "fixed semantic program",
                "policy": numerical["policy"], "observation": OBSERVATION}
    if change:
        metadata.update(change)
    value = tensor if tensor is not None else ("F32", [2], struct.pack("<ff", 1, 0))
    encoded = encode(metadata, dict.fromkeys(GRADIENTS, value))
    path = directory / f"{name}.safetensors"
    path.write_bytes(encoded)
    consumed = {"stage": "consumed", "binding": bound,
                "program": "fixed semantic program", "request": numerical}
    result = {"stage": "result", "binding": bound, "request": numerical,
              "gradients": hashlib.sha256(encoded).hexdigest()}
    log = directory / f"{name}.jsonl"
    log.write_text("\n".join(json.dumps(item) for item in (consumed, result)) + "\n", encoding="utf-8")
    return Input(gradients=path, log=log, call=call)


def replace_tensors(observation, tensors):
    original = observation.gradients.read_bytes()
    length = int.from_bytes(original[:HEADER_BYTES], "little")
    header = json.loads(original[HEADER_BYTES:HEADER_BYTES + length])
    encoded = encode(header["__metadata__"], tensors)
    observation.gradients.write_bytes(encoded)
    events = [json.loads(line) for line in observation.log.read_text().splitlines()]
    events[-1]["gradients"] = hashlib.sha256(encoded).hexdigest()
    observation.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def replace_requests(observation, change, *, stages=("consumed", "result")):
    events = [json.loads(line) for line in observation.log.read_text().splitlines()]
    changed = [{**event, "request": change(event["request"])} if event["stage"] in stages else event
               for event in events]
    observation.log.write_text("\n".join(json.dumps(event) for event in changed) + "\n")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-comparison-"))
        self.policy = self.directory / "policy.safetensors"
        save_file({f"{name}.weight": torch.tensor([1.0, 0.0]) for name in PARAMETERS}, self.policy)
        self.left = fixture(self.directory, "left", call=7)

    def test_distinct_bindings_do_not_change_tensor_equality(self):
        right = fixture(self.directory, "right", call=8)
        result = compare(self.left, right, self.policy)
        self.assertTrue(result["equal"])
        self.assertNotEqual(result["left_digest"], result["right_digest"])
        self.assertNotEqual(result["left_binding"], result["right_binding"])
        self.assertEqual(result["left_tensors"], 4)
        self.assertEqual(result["differences"], [])

    def test_delivery_permutation_preserves_the_declared_numerical_input(self):
        right = fixture(self.directory, "reordered", call=8)
        replace_requests(right, lambda value: {**value, "samples": value["samples"][::-1]})
        before = right.log.read_bytes()
        self.assertTrue(compare(self.left, right, self.policy)["equal"])
        self.assertEqual(right.log.read_bytes(), before)

    def test_delivery_permutation_cannot_hide_algorithm_or_sample_changes(self):
        changes = [lambda value: {**value, "order": value["order"][::-1]},
                   lambda value: {**value, "samples": [{**item, "seed": item["seed"] + 1}
                                                       for item in value["samples"]]},
                   lambda value: {**value, "samples": [{**item, "reward": -item["reward"]}
                                                       for item in value["samples"]]}]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                right = fixture(self.directory, f"reordered{index}", call=8)
                replace_requests(right, lambda value: change({**value, "samples": value["samples"][::-1]}))
                with self.assertRaisesRegex(ValueError, "same program and consumed numerical input"):
                    compare(self.left, right, self.policy)

    def test_result_must_retain_its_own_consumed_delivery(self):
        right = fixture(self.directory, "mismatched", call=8)
        replace_requests(right, lambda value: {**value, "samples": value["samples"][::-1]},
                         stages=("result",))
        with self.assertRaisesRegex(ValueError, "Update result input differs from consumption"):
            compare(self.left, right, self.policy)

    def test_word_differences_are_reported_by_name(self):
        right = fixture(self.directory, "words", call=8, tensor=("F32", [2], struct.pack("<ff", 1, -0.0)))
        result = compare(self.left, right, self.policy)
        self.assertFalse(result["equal"])
        self.assertEqual(result["differences"], [{"tensor": name, "fields": ["data"]} for name in GRADIENTS])

    def test_tensor_metadata_must_match_the_input_adapter(self):
        variants = [("F32", [1, 2], struct.pack("<ff", 1, 0)),
                    ("I32", [2], struct.pack("<ff", 1, 0))]
        for index, tensor in enumerate(variants):
            with self.subTest(tensor=tensor):
                right = fixture(self.directory, f"right{index}", call=8, tensor=tensor)
                with self.assertRaisesRegex(ValueError, "metadata differs from the input adapter"):
                    compare(self.left, right, self.policy)

    def test_matching_digest_cannot_launder_wrong_metadata(self):
        variants = [{"policy": "f" * 64}, {"program": "another program"},
                    {"binding": json.dumps({"call": 99, "attempt": 8, "instance": 8})},
                    {"observation": "after AdamW"}]
        for index, change in enumerate(variants):
            with self.subTest(change=change):
                right = fixture(self.directory, f"wrong{index}", call=8, change=change)
                with self.assertRaisesRegex(ValueError, "Gradient metadata|Gradient observation"):
                    compare(self.left, right, self.policy)

    def test_changed_file_and_mismatched_consumption_are_rejected(self):
        right = fixture(self.directory, "changed", call=8)
        right.gradients.write_bytes(right.gradients.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "reported digest"):
            compare(self.left, right, self.policy)
        right = fixture(self.directory, "input", call=8)
        events = [json.loads(line) for line in right.log.read_text().splitlines()]
        for event in events:
            event["request"]["penalty"] = 0.05
        right.log.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        with self.assertRaisesRegex(ValueError, "same program and consumed numerical input"):
            compare(self.left, right, self.policy)

    def test_missing_role_is_not_an_equal_empty_observation(self):
        right = fixture(self.directory, "missing", call=8)
        replace_tensors(right, {GRADIENTS[0]: ("F32", [2], struct.pack("<ff", 1, 0))})
        with self.assertRaisesRegex(ValueError, "same nonempty parameter set"):
            compare(self.left, right, self.policy)

    def test_matching_omissions_and_extras_cannot_pass_as_equal(self):
        value = ("F32", [2], struct.pack("<ff", 1, 0))
        missing = {name: value for name in GRADIENTS if "lora_B" not in name}
        extra = {**dict.fromkeys(GRADIENTS, value),
                 "objective/unexpected": value, "reward/unexpected": value}
        for index, tensors in enumerate((missing, extra)):
            with self.subTest(index=index):
                left = fixture(self.directory, f"left{index}", call=7)
                right = fixture(self.directory, f"right{index}", call=8)
                replace_tensors(left, tensors)
                replace_tensors(right, tensors)
                with self.assertRaisesRegex(ValueError, "Gradient parameter inventory mismatch"):
                    compare(left, right, self.policy)

    def test_inventory_is_bound_to_consumed_policy_content(self):
        right = fixture(self.directory, "right", call=8)
        changed = self.directory / "changed.safetensors"
        state = load_file(self.policy)
        state[next(iter(state))].add_(1)
        save_file(state, changed)
        with self.assertRaisesRegex(ValueError, "requested tensor identity"):
            compare(self.left, right, changed)

    def test_actual_update_snapshot_matches_input_adapter_inventory(self):
        from learning import update
        from .learning import batch, make_learner
        from probe import adapter_state

        learner = make_learner()
        state = adapter_state(learner.model)
        save_file(state, self.policy)
        expected = inventory(self.policy, digest(state))
        result = update(learner, batch())
        tensors = tuple((name, Tensor(dtype="F32", shape=tuple(value.shape),
                                     data=value.view(torch.uint8).numpy().tobytes()))
                        for name, value in result.gradients.items())
        complete(tensors, expected)
        self.assertEqual(len(tensors), 2 * len(state))
        self.assertGreater(result.summary["gradient_norm"], 0)
        self.assertNotEqual(result.summary["before"], result.summary["after"])

    def test_ambiguous_and_incomplete_log_pairs_are_rejected(self):
        for name in ("duplicate", "incomplete", "truncated"):
            with self.subTest(name=name):
                right = fixture(self.directory, name, call=8)
                lines = right.log.read_text().splitlines()
                text = {"duplicate": "\n".join(lines + lines),
                        "incomplete": lines[0], "truncated": "\n".join(lines) + "\n{"}[name]
                right.log.write_text(text)
                with self.assertRaises(ValueError):
                    compare(self.left, right, self.policy)

    def test_cli_exits_nonzero_for_a_real_byte_difference(self):
        right = fixture(self.directory, "cli", call=8, tensor=("F32", [2], struct.pack("<ff", 1, -0.0)))
        command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "compare.py"), "--policy", str(self.policy)]
        for name, value in (("left", self.left), ("right", right)):
            command += [f"--{name}-gradients", str(value.gradients), f"--{name}-log", str(value.log),
                        f"--{name}-call", str(value.call)]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(result.stdout)["equal"])


if __name__ == "__main__":
    unittest.main()
