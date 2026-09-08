import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .cohort import request
from .compare import replace_requests
from learning import update
from .learning import batch, make_learner
from objective import Reward
from .tokenization import make_tokenizer
from operation import digest as tokenizer_digest
from probe import adapter_state, checkpoint, digest, restore
from states import Input, compare
from update import observation

SEED = 17


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(directory, name, *, call):
    initial = directory / "initial"
    learner = make_learner()
    restore(learner.model, learner.optimizer, initial, tokenizer=make_tokenizer())
    materialization = torch.load(initial / "learner.pt", weights_only=True)
    logical = batch()
    numerical = {**request(), "policy": digest(adapter_state(learner.model)),
                 "learner": sha(initial / "learner.pt"), "reference": digest(adapter_state(learner.model)),
                 "tokenizer": tokenizer_digest(make_tokenizer()),
                 "base": materialization["base"], "assembly": materialization["assembly"],
                 "samples": [observation(item.trajectory, Reward(sample=item.trajectory.request.sample,
                             group=item.trajectory.request.group, value=item.advantage)) for item in logical.samples],
                 "order": list(logical.order), "penalty": logical.profile.penalty}
    result = update(learner, logical)
    output = directory / name
    output.mkdir()
    policy = checkpoint(learner.model, learner.optimizer, output, tokenizer=make_tokenizer(), expected=None)
    save_file(result.gradients, output / "gradients.safetensors")
    bound = {"call": call, "attempt": call, "instance": call}
    consumed = {"stage": "consumed", "binding": bound, "program": "checkpoint association fixture", "request": numerical}
    finished = {"stage": "result", "binding": bound, "request": numerical, "update": result.summary,
                "adapter": digest(policy), "learner": sha(output / "learner.pt"),
                "gradients": sha(output / "gradients.safetensors"), "storage": "staged; not published"}
    log = directory / f"{name}.jsonl"
    log.write_text("\n".join(json.dumps(event) for event in (consumed, finished)) + "\n")
    return Input(checkpoint=output, log=log, call=call)


def report(source, changes):
    events = [json.loads(line) for line in source.log.read_text().splitlines()]
    updated = [events[0], {**events[1], **changes}]
    source.log.write_text("\n".join(json.dumps(event) for event in updated) + "\n")


def rewrite(source, change):
    path = source.checkpoint / "learner.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    stream = io.BytesIO()
    torch.save(change(state), stream)
    path.write_bytes(stream.getvalue())
    report(source, {"learner": sha(path)})


def rewrite_policy(source, change):
    path = source.checkpoint / "adapter.safetensors"
    state = change(load_file(path))
    save_file(state, path)
    events = [json.loads(line) for line in source.log.read_text().splitlines()]
    report(source, {"adapter": digest(state), "update": {**events[-1]["update"], "after": digest(state)}})
    rewrite(source, lambda learner: {**learner, "adapter": digest(state)})


def slot(state, change):
    optimizer = state["optimizer"]
    entries = optimizer["state"]
    key = next(iter(entries))
    return {**state, "optimizer": {**optimizer, "state": {**entries, key: change(entries[key])}}}


def reindex(state):
    names = state["parameters"]
    indices = dict(zip(names, reversed(names), strict=True))
    optimizer = state["optimizer"]
    groups = [{**group, "params": [indices[key] for key in group["params"]]} for group in optimizer["param_groups"]]
    return {**state, "parameters": {indices[key]: name for key, name in names.items()},
            "optimizer": {**optimizer, "param_groups": groups,
                          "state": {indices[key]: value for key, value in optimizer["state"].items()}}}


class StateTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-states-"))
        initial = self.directory / "initial"
        initial.mkdir()
        torch.manual_seed(SEED)
        learner = make_learner()
        checkpoint(learner.model, learner.optimizer, initial, tokenizer=make_tokenizer(), expected=None)
        self.policy = initial / "adapter.safetensors"
        self.left = fixture(self.directory, "left", call=7)

    def test_actual_checkpoint_values_equal_across_bindings_and_container_bytes(self):
        right = fixture(self.directory, "right", call=8)
        rewrite(right, lambda state: state)
        replace_requests(right, lambda value: {**value, "samples": value["samples"][::-1]})
        rng = torch.get_rng_state()
        result = compare(self.left, right, self.policy)
        self.assertTrue(result["equal"])
        self.assertNotEqual(result["left_learner"], result["right_learner"])
        self.assertEqual(result["differences"], [])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def changed_moment(self, name):
        right = fixture(self.directory, name, call=8)
        rewrite(right, lambda state: slot(state, lambda value: {**value, "exp_avg": -value["exp_avg"]}))
        return right

    def test_signed_zero_moment_change_is_visible_without_a_policy_change(self):
        right = self.changed_moment("signed")
        result = compare(self.left, right, self.policy)
        self.assertFalse(result["equal"])
        self.assertTrue(result["policy_equal"])
        self.assertFalse(result["learner_equal"])
        name = torch.load(right.checkpoint / "learner.pt", weights_only=True)["parameters"][0]
        self.assertEqual(result["differences"],
                         [{"path": ["learner", "optimizer", "state", name, "exp_avg"], "fields": ["data"]}])

    def test_rng_bytes_are_part_of_the_learner_observation(self):
        right = fixture(self.directory, "rng", call=8)
        rewrite(right, lambda state: {**state, "cpu_rng": state["cpu_rng"] ^ 1})
        result = compare(self.left, right, self.policy)
        self.assertEqual(result["differences"], [{"path": ["learner", "cpu_rng"], "fields": ["data"]}])

    def test_parameter_renumbering_preserves_named_state_observations(self):
        right = fixture(self.directory, "renumbered", call=8)
        rewrite(right, reindex)
        self.assertTrue(compare(self.left, right, self.policy)["equal"])

    def test_output_materialization_must_match_the_consumed_input(self):
        for field in ("base", "assembly"):
            with self.subTest(field=field):
                right = fixture(self.directory, field, call=8)
                rewrite(right, lambda state: {**state, field: "f" * 64})
                with self.assertRaisesRegex(ValueError, "materialization differs from the consumed update input"):
                    compare(self.left, right, self.policy)

    def test_matching_missing_parameter_bindings_cannot_pass(self):
        right = fixture(self.directory, "unbound", call=8)
        for source in (self.left, right):
            rewrite(source, lambda state: {**state, "parameters": {}})
        with self.assertRaisesRegex(ValueError, "parameter binding inventory"):
            compare(self.left, right, self.policy)

    def test_checkpoint_tokenizer_must_retain_the_consumed_identity(self):
        right = fixture(self.directory, "tokenizer", call=8)
        rewrite(right, lambda state: {**state, "tokenizer": "f" * 64})
        with self.assertRaisesRegex(ValueError, "tokenizer differs from the consumed update input"):
            compare(self.left, right, self.policy)

    def test_file_digest_and_checkpoint_pair_binding_are_required(self):
        changed = fixture(self.directory, "bytes", call=8)
        path = changed.checkpoint / "learner.pt"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "reported digest"):
            compare(self.left, changed, self.policy)
        mixed = fixture(self.directory, "mixed", call=8)
        rewrite(mixed, lambda state: {**state, "adapter": "f" * 64})
        with self.assertRaisesRegex(ValueError, "adapter binding mismatch"):
            compare(self.left, mixed, self.policy)

    def test_matching_missing_slots_and_wrong_shapes_cannot_pass(self):
        changes = [lambda state: slot(state, lambda value: {name: item for name, item in value.items() if name != "exp_avg"}),
                   lambda state: {**state, "optimizer": {**state["optimizer"], "state": {}}},
                   lambda state: slot(state, lambda value: {**value, "exp_avg": value["exp_avg"].reshape(-1),
                                                           "exp_avg_sq": value["exp_avg_sq"].reshape(-1)})]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                left = fixture(self.directory, f"missing-left{index}", call=7)
                right = fixture(self.directory, f"missing-right{index}", call=8)
                rewrite(left, change)
                rewrite(right, change)
                with self.assertRaisesRegex(ValueError, "AdamW slot fields|slot inventory|moment inventory"):
                    compare(left, right, self.policy)

    def test_matching_output_policy_omission_cannot_hide_behind_its_digest(self):
        right = fixture(self.directory, "omitted", call=8)
        for source in (self.left, right):
            rewrite_policy(source, lambda state: dict(list(state.items())[:-1]))
        with self.assertRaisesRegex(ValueError, "Adapter keys do not match"):
            compare(self.left, right, self.policy)

    def test_output_policy_tensor_differences_are_reported(self):
        right = fixture(self.directory, "policy", call=8)
        name = next(iter(load_file(right.checkpoint / "adapter.safetensors")))
        rewrite_policy(right, lambda state: {**state, name: state[name] + 1})
        result = compare(self.left, right, self.policy)
        self.assertFalse(result["policy_equal"])
        self.assertFalse(result["equal"])
        self.assertIn({"path": ["policy", name], "fields": ["data"]}, result["differences"])

    def test_optimizer_group_must_match_the_declared_profile_and_inventory(self):
        changes = [{"lr": 0.5}, {"decoupled_weight_decay": False},
                   {"decoupled_weight_decay": 1}, {"params": [0, 0]}]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                right = fixture(self.directory, f"group{index}", call=8)
                rewrite(right, lambda state: {**state, "optimizer": {**state["optimizer"],
                        "param_groups": [{**state["optimizer"]["param_groups"][0], **change}]}})
                with self.assertRaisesRegex(ValueError, "AdamW settings|parameter inventory"):
                    compare(self.left, right, self.policy)

    def test_invalid_state_format_is_not_a_comparable_learner(self):
        changes = [lambda state: {name: value for name, value in state.items() if name != "cpu_rng"},
                   lambda state: {**state, "base": "invalid"},
                   lambda state: {**state, "assembly": "invalid"},
                   lambda state: {**state, "cpu_rng": state["cpu_rng"].float()},
                   lambda state: slot(state, lambda value: {**value, "step": torch.tensor(0.0)}),
                   lambda state: slot(state, lambda value: {**value, "exp_avg": value["exp_avg"].double()}),
                   lambda state: slot(state, lambda value: {**value, "exp_avg_sq": value["exp_avg_sq"] + float("inf")})]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                right = fixture(self.directory, f"invalid{index}", call=8)
                rewrite(right, change)
                with self.assertRaises(ValueError):
                    compare(self.left, right, self.policy)

    def test_wrong_input_adapter_and_changed_algorithm_are_rejected(self):
        right = fixture(self.directory, "algorithm", call=8)
        with self.assertRaisesRegex(ValueError, "requested tensor identity"):
            compare(self.left, right, right.checkpoint / "adapter.safetensors")
        replace_requests(right, lambda value: {**value, "order": value["order"][::-1]})
        with self.assertRaisesRegex(ValueError, "same program and consumed numerical input"):
            compare(self.left, right, self.policy)

    def test_cli_reports_an_actual_optimizer_byte_difference(self):
        right = self.changed_moment("cli")
        command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "states.py"), "--policy", str(self.policy)]
        for name, value in (("left", self.left), ("right", right)):
            command += [f"--{name}-checkpoint", str(value.checkpoint), f"--{name}-log", str(value.log),
                        f"--{name}-call", str(value.call)]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(result.stdout)["learner_equal"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
