import hashlib
from pathlib import Path

FORMAT = "invar-worker-source-v1"
SOURCE_FILES = (
    "assembly.py", "backend.py", "binding.py", "cohort.py", "experiment.py",
    "frozen.py", "implementation.py", "infer.py", "session.py", "inference.py", "initialize.py", "invocation.py",
    "learning.py", "registry.py", "metrics.py", "objective.py", "policy.py", "probability.py", "probe.py",
    "rollout.py", "step.py", "tensors.py", "operation.py", "tokenization.py", "update.py",
)


def description(root):
    return {"format": FORMAT, "files": {name: file_digest(root / name) for name in SOURCE_FILES}}


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def current():
    return description(Path(__file__).resolve().parent)
