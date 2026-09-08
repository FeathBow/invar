import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cohort import fields, identity, number

WORKER_STAGES = frozenset(("loading", "profile", "load", "unloaded_adapter", "loaded_adapter", "consumed", "inference", "result"))


@dataclass(frozen=True, kw_only=True)
class Run:
    log: Path
    policy: str
    exit_code: int


@dataclass(frozen=True, kw_only=True)
class Input:
    digest: str
    cohorts: tuple


@dataclass(frozen=True, kw_only=True)
class Sample:
    cohort: int
    name: str
    group: str
    seed: int
    reward: int
    response_tokens: int
    truncated: bool


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate JSON fields")
    return result


def decode(encoded):
    return json.loads(encoded, object_pairs_hook=unique)


def snapshot(path):
    encoded = path.read_bytes()
    return hashlib.sha256(encoded).hexdigest(), encoded


def natural(value):
    require(type(value) is int and value >= 0, "Expected a nonnegative integer count or identity")
    return value


def declared(task):
    fields(task, "name group prompt seed tokens temperature answer")
    require(all(isinstance(task[name], str) and task[name] for name in ("name", "group", "prompt", "answer")),
            "Evaluation task names, groups, prompts and answers must be nonempty text")
    require(type(task["seed"]) is int, "Expected an integral evaluation seed")
    require(natural(task["tokens"]) > 0 and number(task["temperature"]) > 0, "Invalid evaluation token budget or temperature")
    return task


def tasks(path):
    digest, encoded = snapshot(path)
    cycles = decode(encoded)
    require(isinstance(cycles, list) and cycles, "Expected nonempty evaluation cohorts")
    return Input(digest=digest, cohorts=tuple(definition(cycle) for cycle in cycles))


def definition(cycle):
    fields(cycle, "tasks order delivery")
    require(isinstance(cycle["tasks"], list) and cycle["tasks"], "Expected nonempty evaluation tasks")
    values = tuple(declared(task) for task in cycle["tasks"])
    require(len({task["name"] for task in values}) == len(values), "Repeated evaluation sample name")
    require(all(count >= 2 for count in Counter(task["group"] for task in values).values()),
            "Evaluation groups require at least two samples")
    for axis in ("order", "delivery"):
        order = cycle[axis]
        require(isinstance(order, list) and sorted(natural(index) for index in order) == list(range(len(values))),
                "Invalid evaluation execution or delivery permutation")
    return values


def sample(value, task, index):
    fields(value, "name group seed reward response_tokens truncated binding")
    require(value["name"] == task["name"] and value["group"] == task["group"]
            and type(value["seed"]) is int and value["seed"] == task["seed"], "Evaluation sample differs from its declared task")
    reward = number(value["reward"])
    count = natural(value["response_tokens"])
    require(reward in (0, 1), "Expected the binary decimal-answer reward profile")
    require(0 < count <= task["tokens"], "Evaluation response length exceeds its declared budget")
    require(type(value["truncated"]) is bool, "Expected a Boolean truncation flag")
    require(not value["truncated"] or (count == task["tokens"] and reward == 0), "Truncated evaluation sample has invalid length or reward")
    return Sample(cohort=index, name=task["name"], group=task["group"], seed=task["seed"],
                  reward=int(reward), response_tokens=count, truncated=value["truncated"])


def binding(value):
    fields(value, "call attempt instance")
    return tuple(natural(value[name]) for name in ("call", "attempt", "instance"))


def summary(samples):
    groups = {}
    for item in samples:
        groups.setdefault(item.group, set()).add(item.reward)
    return {"sample_count": len(samples), "reward_sum": sum(item.reward for item in samples),
            "response_tokens": sum(item.response_tokens for item in samples),
            "truncated_count": sum(item.truncated for item in samples), "group_count": len(groups),
            "zero_variance_groups": sum(len(rewards) == 1 for rewards in groups.values())}


def check_summary(actual, expected):
    fields(actual, "sample_count reward_sum response_tokens truncated_count group_count zero_variance_groups")
    for name, value in expected.items():
        observed = number(actual[name]) if name == "reward_sum" else natural(actual[name])
        require(observed == value, "Evaluation summary differs from its sample records")


def cohort(value, expected, index):
    fields(value, "phase cohort policy summary samples")
    require(value["phase"] == "evaluation" and natural(value["cohort"]) == index, "Missing or reordered evaluation cohort")
    supplied = value["samples"]
    require(isinstance(supplied, list) and len(supplied) == len(expected), "Incomplete evaluation sample inventory")
    by_name = {item["name"]: item for item in supplied}
    require(len(by_name) == len(supplied) and set(by_name) == {task["name"] for task in expected},
            "Evaluation sample names differ from the declared cohort")
    samples = tuple(sample(by_name[task["name"]], task, index) for task in expected)
    check_summary(value["summary"], summary(samples))
    return samples, tuple(binding(item["binding"]) for item in supplied)


def completed(value, policy, expected, loads):
    declared = " sessions" if "sessions" in value else ""
    fields(value, "phase policy cohorts tasks_sha256" + declared + " " + " ".join(model_binding(value)))
    require(value["phase"] == "evaluation_complete" and value["policy"] == policy
            and natural(value["cohorts"]) == len(expected.cohorts), "Missing or mismatched evaluation completion")
    require(value["tasks_sha256"] == expected.digest, "Evaluation input identity differs from the supplied frozen bytes")
    if declared:
        sessions = natural(value["sessions"])
        require(sessions >= 1 and all(count == sessions for count in loads),
                "Declared session count differs from the recorded model loads of a cohort")


def model_binding(value):
    materialized = "base" in value or "assembly" in value
    if materialized:
        names = ("tokenizer", "base", "assembly")
        require(all(name in value for name in names), "Incomplete model materialization binding")
    else:
        names = ("tokenizer",) if "tokenizer" in value else ()
    return {name: identity(value[name]) for name in names}


def model_records(records):
    expected = model_binding(records[-1])
    for record in records:
        if record.get("stage") in ("loaded_adapter", "consumed", "result"):
            require(model_binding(record) == expected, "Evaluation model and tokenizer bindings disagree")


def evaluation_records(encoded):
    records = tuple(decode(line) for line in encoded.splitlines())
    require(bool(records), "Missing evaluation records")
    for record in records:
        require(isinstance(record, dict), "Expected JSON objects in evaluation output")
    require(records[-1].get("phase") == "evaluation_complete", "Missing final evaluation completion or trailing output")
    model_records(records)
    loads, current = [], 0
    for record in records[:-1]:
        if "phase" in record:
            require(record["phase"] == "evaluation" and "stage" not in record, "Unexpected evaluation record")
            loads.append(current)
            current = 0
        else:
            require(record.get("stage") in WORKER_STAGES, "Unknown worker record in evaluation output")
            current += record["stage"] == "load"
    require(current == 0, "Model load recorded outside every evaluation cohort")
    return tuple(record for record in records if "phase" in record), tuple(loads)


def read(run, expected):
    identity(run.policy)
    require(type(run.exit_code) is int and run.exit_code == 0, "Evaluation process did not exit successfully")
    digest, encoded = snapshot(run.log)
    require(encoded.endswith(b"\n"), "Incomplete final evaluation line")
    records, loads = evaluation_records(encoded)
    require(len(records) == len(expected.cohorts) + 1, "Incomplete or trailing evaluation records")
    completed(records[-1], run.policy, expected, loads)
    samples, bindings = [], []
    for index, (record, tasks) in enumerate(zip(records[:-1], expected.cohorts, strict=True)):
        require(record["policy"] == run.policy, "Evaluation cohort policy mismatch")
        observed, used = cohort(record, tasks, index)
        samples.extend(observed)
        bindings.extend(used)
    require(all(len({value[axis] for value in bindings}) == len(bindings) for axis in range(3)),
            "Evaluation reused a call, attempt or instance identity")
    return digest, tuple(samples)
