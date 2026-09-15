import hashlib
from dataclasses import dataclass
from pathlib import Path

from worker import core as host
from worker.cohort import fields, identity
from worker.core import decode

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
    path: Path
    core: str


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


def snapshot(path):
    encoded = path.read_bytes()
    return hashlib.sha256(encoded).hexdigest(), encoded


def natural(value):
    require(type(value) is int and value >= 0, "Expected a nonnegative integer count or identity")
    return value


def tasks(path, *, core="invar"):
    observed = host.invoke(("inspect", "tasks", "--input", path), executable=core)
    fields(observed, "digest cohorts")
    require(isinstance(observed["cohorts"], list)
            and all(isinstance(cycle, list) for cycle in observed["cohorts"]), "Invalid core workload response")
    cohorts = tuple(tuple(fields(task, "name group prompt seed tokens temperature answer") for task in cycle)
                    for cycle in observed["cohorts"])
    return Input(digest=identity(observed["digest"]), cohorts=cohorts, path=path, core=core)


def sample(value):
    fields(value, "cohort name group seed reward response_tokens truncated binding")
    require(all(isinstance(value[key], str) for key in ("name", "group"))
            and type(value["seed"]) is int and type(value["truncated"]) is bool, "Invalid core sample response")
    for key in ("cohort", "reward", "response_tokens"):
        natural(value[key])
    binding(value["binding"])
    return Sample(**{name: value[name] for name in Sample.__dataclass_fields__})


def binding(value):
    fields(value, "call attempt instance")
    return tuple(natural(value[name]) for name in ("call", "attempt", "instance"))


def model_binding(value):
    materialized = "base" in value or "assembly" in value
    if materialized:
        names = ("tokenizer", "base", "assembly")
        require(all(name in value for name in names), "Incomplete model materialization binding")
    else:
        names = ("tokenizer",) if "tokenizer" in value else ()
    return {name: identity(value[name]) for name in names}


def read(run, expected):
    observed = host.invoke(("inspect", "evaluation", "--tasks", expected.path, "--tasks-digest", expected.digest,
                            "--log", run.log, "--policy", run.policy, "--exit-code", run.exit_code), executable=expected.core)
    fields(observed, "format tasks_sha256 log_sha256 policy model samples")
    require(observed["format"] == "invar-evaluation-report-v1" and observed["tasks_sha256"] == expected.digest
            and observed["policy"] == run.policy, "Mismatched core evaluation response")
    model_binding(observed["model"])
    require(isinstance(observed["samples"], list), "Invalid core evaluation sample sequence")
    return identity(observed["log_sha256"]), tuple(sample(value) for value in observed["samples"])
