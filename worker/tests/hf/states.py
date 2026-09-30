import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import hashlib
import io
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from worker.tests.hf.cohort import request
from worker.hf import codec
from worker import core
from worker.hf.learning import update
from worker.tests.hf.learning import ADVANTAGES, LOGICAL_ORDER, batch, make_learner
from worker.tests.hf.tokenization import make_tokenizer
from worker.hf.operation import digest as tokenizer_digest
from worker.hf.model import adapter_state
from worker.hf.checkpoint import checkpoint, restore
from worker.hf.tensors import digest
from worker.update import observation

SEED = 17


@dataclass(frozen=True, kw_only=True)
class Input:
    checkpoint: Path
    log: Path
    call: int


def inputs(left, right, kind):
    result = []
    for side, value in (("left", left), ("right", right)):
        result.extend((f"--{side}-log", value.log, f"--{side}-call", value.call, f"--{side}-{kind}", getattr(value, kind)))
    return result


def compare(left, right, policy, *, core_executable="invar"):
    return through_codec(left, right, policy, codec.Session().handle, core_executable=core_executable)


def replace_requests(observation, change, *, stages=("consumed", "result")):
    events = [json.loads(line) for line in observation.log.read_text().splitlines()]
    changed = [{**event, "request": change(event["request"])} if event["stage"] in stages else event
               for event in events]
    observation.log.write_text("\n".join(json.dumps(event) for event in changed) + "\n")


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
                 "samples": [observation(item, SimpleNamespace(sample=name, group=item.request.group, reward=reward, version=0,
                                                               behavior_policy=digest(adapter_state(learner.model)),
                                                               reference_bits=(), advantage_bits=advantage))
                             for (name, item), (_, _, advantage), reward in zip(logical.trajectories.items(),
                                                                                logical.exchange.samples.values(), ADVANTAGES, strict=True)],
                 "order": list(LOGICAL_ORDER), "steps": [list(batch) for batch in logical.steps],
                 "penalty": logical.exchange.profile.penalty}
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
    return renumber(state, indices)


def tensor_descriptions(value):
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            return [value]
        return [item for child in value.values() for item in tensor_descriptions(child)]
    if isinstance(value, list):
        return [item for child in value for item in tensor_descriptions(child)]
    return []


def through_codec(left, right, policy, handler, *, core_executable="invar"):
    arguments = ["compare", "states", "--codec-mode", "stdio", "--policy", policy,
                 *inputs(left, right, "checkpoint")]
    return core.exchange(arguments, executable=core_executable, handler=handler)


def renumber(state, indices):
    names = state["parameters"]
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

    def test_selected_core_is_required_without_a_fallback(self):
        executable = shutil.which("invar")
        self.assertIsNotNone(executable, "Build invar and add it to this test process's PATH")
        right = fixture(self.directory, "selected", call=8)
        self.assertEqual(compare(self.left, right, self.policy, core_executable=executable),
                         compare(self.left, right, self.policy))
        with self.assertRaises(FileNotFoundError):
            compare(self.left, right, self.policy, core_executable=self.directory / "missing-invar")

    def test_native_cli_uses_the_actual_external_serialization_codec(self):
        right = fixture(self.directory, "external", call=8)
        entries = Path(__file__).resolve().parents[3] / "entries"
        command = ["invar", "compare", "states", "--python", sys.executable,
                   "--codec", str(entries / "codec.py"), "--policy", str(self.policy)]
        for name, value in (("left", self.left), ("right", right)):
            command += [f"--{name}-checkpoint", str(value.checkpoint), f"--{name}-log", str(value.log),
                        f"--{name}-call", str(value.call)]
        result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parents[3], timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), compare(self.left, right, self.policy))

    def test_scoped_codec_releases_snapshots_after_each_actual_comparison(self):
        right = fixture(self.directory, "scoped-codec", call=8)
        session = codec.Session()
        for _ in range(2):
            result = through_codec(self.left, right, self.policy, session.handle)
            self.assertTrue(result["equal"])
            self.assertEqual(session.tensors, [])
            self.assertEqual(session.views, {})

    def test_core_rejects_corrupted_descriptions_from_actual_native_decoding(self):
        mutations = [lambda value: value.update(source_sha256="f" * 64),
                     lambda value: tensor_descriptions(value)[1].update(index=tensor_descriptions(value)[0]["index"]),
                     lambda value: tensor_descriptions(value)[0].update(size=0),
                     lambda value: tensor_descriptions(value)[0].update(type="")]
        right = fixture(self.directory, "codec-description", call=8)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                session = codec.Session()

                def handler(request, target):
                    if request["codec"] == "decode":
                        value = session.decode(request["path"])
                        mutation(value)
                        codec.write(target, json.dumps(value).encode() + b"\n")
                        target.flush()
                    else:
                        session.handle(request, target)

                with self.assertRaises(ValueError):
                    through_codec(self.left, right, self.policy, handler)

    def test_tensor_reference_reuse_cannot_hide_a_changed_checkpoint(self):
        right = self.changed_moment("codec-alias")
        session = codec.Session()

        def handler(request, target):
            if request["codec"] == "decode":
                offset = len(session.tensors)
                value = session.decode(request["path"])
                for tensor in tensor_descriptions(value):
                    tensor["index"] -= offset
                codec.write(target, json.dumps(value).encode() + b"\n")
                target.flush()
            else:
                session.handle(request, target)

        with self.assertRaisesRegex(ValueError, "reused across checkpoint snapshots"):
            through_codec(self.left, right, self.policy, handler)

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

    def test_native_parameter_ids_larger_than_machine_integers_are_preserved(self):
        right = fixture(self.directory, "large-identities", call=8)
        offset = 1 << 1024
        rewrite(right, lambda state: renumber(state, {key: offset + index
                                                     for index, key in enumerate(state["parameters"])}))
        self.assertTrue(compare(self.left, right, self.policy)["equal"])

    def test_unsupported_native_integer_encoding_fails_without_an_unsafe_fallback(self):
        right = fixture(self.directory, "unsupported-identities", call=8)
        offset = 1 << 16000
        rewrite(right, lambda state: renumber(state, {key: offset + index
                                                     for index, key in enumerate(state["parameters"])}))
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported operand"):
            compare(self.left, right, self.policy)

    def test_scalar_types_and_signed_zero_remain_observable(self):
        rewrite(self.left, lambda state: {**state, "optimizer": {**state["optimizer"],
                "param_groups": [{**state["optimizer"]["param_groups"][0], "weight_decay": 0.0}]}})
        for label, value, field in (("integer", 0, "type"), ("negative-zero", -0.0, "value")):
            with self.subTest(label=label):
                right = fixture(self.directory, label, call=8)
                rewrite(right, lambda state: {**state, "optimizer": {**state["optimizer"],
                        "param_groups": [{**state["optimizer"]["param_groups"][0], "weight_decay": value}]}})
                self.assertEqual(compare(self.left, right, self.policy)["differences"],
                                 [{"path": ["learner", "optimizer", "param_groups", 0, "weight_decay"], "fields": [field]}])

    def test_tensor_subclass_identity_remains_observable(self):
        right = fixture(self.directory, "parameter-class", call=8)
        rewrite(right, lambda state: slot(state, lambda value: {
            **value, "exp_avg": torch.nn.Parameter(value["exp_avg"], requires_grad=False)}))
        name = torch.load(right.checkpoint / "learner.pt", weights_only=True)["parameters"][0]
        self.assertEqual(compare(self.left, right, self.policy)["differences"],
                         [{"path": ["learner", "optimizer", "state", name, "exp_avg"], "fields": ["type"]}])

    def test_recorded_cuda_rng_sequence_order_and_length_are_observable(self):
        right = fixture(self.directory, "rng-sequence", call=8)
        for source in (self.left, right):
            rewrite(source, lambda state: {**state, "cuda_rng": [state["cpu_rng"], state["cpu_rng"] ^ 1]})
        self.assertTrue(compare(self.left, right, self.policy)["equal"])
        rewrite(right, lambda state: {**state, "cuda_rng": state["cuda_rng"][::-1]})
        self.assertEqual(compare(self.left, right, self.policy)["differences"],
                         [{"path": ["learner", "cuda_rng", index], "fields": ["data"]} for index in range(2)])
        rewrite(right, lambda state: {**state, "cuda_rng": state["cuda_rng"][:1]})
        self.assertEqual(compare(self.left, right, self.policy)["differences"],
                         [{"path": ["learner", "cuda_rng"], "fields": ["length"]}])

    def test_output_materialization_must_match_the_consumed_input(self):
        for field in ("base", "assembly"):
            with self.subTest(field=field):
                right = fixture(self.directory, field, call=8)
                rewrite(right, lambda state: {**state, field: "f" * 64})
                with self.assertRaisesRegex(ValueError, "materialization differs from the declared input"):
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
        with self.assertRaisesRegex(ValueError, "tokenizer differs from the declared input"):
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


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
