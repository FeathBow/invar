import hashlib
from pathlib import Path

FORMAT = "invar-worker-source-v1"
SOURCE_FILES = (
    "advantage.py",
    "binding.py",
    "cohort.py",
    "implementation.py",
    "inputs.py",
    "invocation.py",
    "learner.py",
    "logical.py",
    "record.py",
    "registry.py",
    "report.py",
    "resident.py",
    "scalar.py",
    "tokenization.py",
    "hf/assembly.py",
    "hf/backend.py",
    "hf/decoding.py",
    "hf/frozen.py",
    "hf/infer.py",
    "hf/inference.py",
    "hf/initialize.py",
    "hf/learning.py",
    "hf/metrics.py",
    "hf/objective.py",
    "hf/operation.py",
    "hf/policy.py",
    "hf/probability.py",
    "hf/probe.py",
    "hf/resident.py",
    "hf/rollout.py",
    "hf/runtime.py",
    "hf/session.py",
    "hf/state.py",
    "hf/step.py",
    "hf/tensors.py",
    "trajectory.py",
    "update.py",
)


def description(root):
    return {"format": FORMAT, "files": {name: file_digest(root / name) for name in SOURCE_FILES}}


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def current():
    return description(Path(__file__).resolve().parent)
