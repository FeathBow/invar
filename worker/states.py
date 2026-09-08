import argparse
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

import cohort
from binding import matches, observation, schema
from compare import canonical, paired
from policy import read_adapter, validate_schema
from tensors import tensor_bytes
from step import optimizer_options
from update import snapshot

SLOTS = {"step", "exp_avg", "exp_avg_sq"}


@dataclass(frozen=True, kw_only=True)
class Input:
    checkpoint: Path
    log: Path
    call: int


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    policy: dict[str, torch.Tensor]
    learner: dict
    policy_digest: str
    learner_digest: str


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identities(claim):
    result = json.loads(claim.result)
    policy = cohort.identity(result["adapter"])
    learner = cohort.identity(result["learner"])
    require(result["storage"] == "staged; not published", "Expected a staged update report")
    require(result["update"]["before"] == json.loads(claim.request)["policy"]
            and result["update"]["after"] == policy, "Update policy identities disagree")
    return policy, learner


def read(source, claim, expected):
    policy_digest, learner_digest = identities(claim)
    policy = read_adapter(source.checkpoint / "adapter.safetensors", policy_digest)
    validate_schema(expected, policy)
    actual, encoded = snapshot(source.checkpoint / "learner.pt")
    require(actual == learner_digest, "Learner file differs from its reported digest")
    learner = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    require(isinstance(learner, dict) and set(learner) == {
        "adapter", "base", "assembly", "tokenizer", "parameters", "optimizer", "cpu_rng", "cuda_rng"},
            "Unexpected learner checkpoint fields")
    require(learner["adapter"] == policy_digest, "Checkpoint adapter binding mismatch")
    cohort.identity(learner["base"])
    cohort.identity(learner["assembly"])
    request = cohort.decode(json.loads(claim.request))
    require(learner["base"] == request.base and learner["assembly"] == request.assembly,
            "Checkpoint model materialization differs from the consumed update input")
    require(cohort.identity(learner["tokenizer"]) == request.tokenizer,
            "Checkpoint tokenizer differs from the consumed update input")
    validate_optimizer(learner, policy, request)
    validate_rng(learner["cpu_rng"])
    require(isinstance(learner["cuda_rng"], list), "Expected a CUDA RNG state list")
    for state in learner["cuda_rng"]:
        validate_rng(state)
    return Checkpoint(policy=policy, learner=observation(learner), policy_digest=policy_digest,
                      learner_digest=learner_digest)


def group_parameters(value, request, count):
    require(isinstance(value, dict) and set(value) == {"state", "param_groups"},
            "Unexpected optimizer checkpoint fields")
    groups = value["param_groups"]
    require(isinstance(groups, list) and len(groups) == 1, "Expected the single-group AdamW profile")
    group = groups[0]
    expected = {**optimizer_options(request.optimizer), "decoupled_weight_decay": True}
    require(isinstance(group, dict) and set(group) == set(expected) | {"params"},
            "Unexpected AdamW parameter group fields")
    require(all(setting(group[key], item) for key, item in expected.items()),
            "AdamW settings differ from the consumed input")
    parameters = group["params"]
    require(isinstance(parameters, list) and all(type(key) is int and key >= 0 for key in parameters),
            "Expected optimizer parameter identities")
    require(len(parameters) == len(set(parameters)) == count, "Optimizer parameter inventory mismatch")
    return parameters


def validate_optimizer(learner, policy, request):
    value = learner["optimizer"]
    parameters = group_parameters(value, request, len(policy))
    names = learner["parameters"]
    expected = schema(policy)
    require(matches(names, expected) and set(names) == set(parameters), "Optimizer parameter binding inventory mismatch")
    slots = value["state"]
    require(isinstance(slots, dict) and all(type(key) is int for key in slots)
            and set(slots) == set(parameters), "Optimizer slot inventory mismatch")
    for key, state in slots.items():
        validate_slots(state)
        require(state["exp_avg"].shape == expected[names[key]].shape, "AdamW moment inventory parameter shape mismatch")


def setting(actual, expected):
    if isinstance(expected, tuple):
        return (isinstance(actual, tuple) and len(actual) == len(expected)
                and all(setting(left, right) for left, right in zip(actual, expected, strict=True)))
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    return type(actual) in (int, float) and actual == expected


def validate_slots(value):
    require(isinstance(value, dict) and set(value) == SLOTS, "Unexpected AdamW slot fields")
    require(all(isinstance(slot, torch.Tensor) and slot.dtype == torch.float32 and slot.isfinite().all()
                for slot in value.values()), "Expected finite FP32 AdamW state tensors")
    step = value["step"]
    require(step.ndim == 0 and step.item() > 0 and step.item().is_integer(),
            "Expected a positive integral post-update AdamW step")
    require(value["exp_avg"].shape == value["exp_avg_sq"].shape and value["exp_avg"].numel() > 0,
            "AdamW moment shapes disagree")


def validate_rng(value):
    require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
            and value.ndim == 1 and value.numel() > 0, "Expected a nonempty RNG byte vector")


def tensor_changes(first, second):
    fields = [name for name in ("dtype", "shape") if getattr(first, name) != getattr(second, name)]
    if not torch.equal(tensor_bytes(first), tensor_bytes(second)):
        fields.append("data")
    return fields


def mapping_changes(first, second, path):
    changes = []
    for key in sorted(first.keys() | second.keys()):
        if key not in first or key not in second:
            changes.append({"path": [*path, key], "missing": "left" if key not in first else "right"})
        else:
            changes.extend(differences(first[key], second[key], (*path, key)))
    return changes


def sequence_changes(first, second, path):
    if len(first) != len(second):
        return [{"path": list(path), "fields": ["length"]}]
    return [change for index, (left, right) in enumerate(zip(first, second, strict=True))
            for change in differences(left, right, (*path, index))]


def differences(first, second, path):
    if type(first) is not type(second):
        return [{"path": list(path), "fields": ["type"]}]
    if isinstance(first, dict):
        return mapping_changes(first, second, path)
    if isinstance(first, (list, tuple)):
        return sequence_changes(first, second, path)
    if isinstance(first, torch.Tensor):
        fields = tensor_changes(first, second)
    else:
        equal = struct.pack("!d", first) == struct.pack("!d", second) if isinstance(first, float) else first == second
        fields = [] if equal else ["value"]
    return [{"path": list(path), "fields": fields}] if fields else []


def compare(left, right, policy):
    claims = paired(left, right)
    expected = read_adapter(policy, json.loads(claims[0].request)["policy"])
    first, second = (read(source, claim, expected) for source, claim in zip((left, right), claims, strict=True))
    policies = differences(first.policy, second.policy, ("policy",))
    learner = differences(first.learner, second.learner, ("learner",))
    return {"comparison": "checkpoint values and tensor bytes", "equal": not policies and not learner,
            "policy_equal": not policies, "learner_equal": not learner,
            "left_policy": first.policy_digest, "right_policy": second.policy_digest,
            "left_learner": first.learner_digest, "right_learner": second.learner_digest,
            "left_binding": claims[0].invocation.binding(), "right_binding": claims[1].invocation.binding(),
            "differences": policies + learner,
            "scope": "reported checkpoint contents; not execution, restoration or numerical qualification"}


def arguments():
    parser = argparse.ArgumentParser(description="Compare bound post-update policy and learner checkpoints")
    parser.add_argument("--policy", type=Path, required=True, help="Actual common input adapter checkpoint")
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{side}-log", type=Path, required=True)
        parser.add_argument(f"--{side}-call", type=int, required=True)
    values = vars(parser.parse_args())
    sides = tuple(Input(**{field: values[f"{side}_{field}"] for field in ("checkpoint", "log", "call")})
                  for side in ("left", "right"))
    return *sides, values["policy"]


def main():
    result = compare(*arguments())
    print(canonical(result))
    raise SystemExit(0 if result["equal"] else 1)


if __name__ == "__main__":
    main()
