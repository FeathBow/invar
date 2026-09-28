import hashlib
from pathlib import Path

FORMAT = "invar-worker-implementation-v2"
INFERENCE = "inference"
LEARNING = "learning"
SHARED = {
    "hf/model.py": "model loading: quantization, dtype, attention implementation and LoRA wrapping",
    "hf/backend.py": "torch numerical switches: matmul precision, TF32 and determinism",
    "hf/policy.py": "adapter tensors installed into the model",
}
ROLES = {
    INFERENCE: {
        **SHARED,
        "hf/inference.py": "runtime construction, device placement and thread count",
        "hf/rollout.py": "sampling loop: temperature, softmax, seeded generator and EOS stop",
        "hf/decoding.py": "forward pass with the KV cache",
        "hf/operation.py": "tokenizer loading: chat template and special tokens",
        "tokenization.py": "prompt token construction",
    },
    LEARNING: {
        **SHARED,
        "hf/checkpoint.py": "optimizer, parameter and random state saved and restored between updates",
        "hf/learning.py": "learner graph, per-step vector-Jacobian products and AdamW steps in the declared order",
        "hf/probability.py": "learner log probabilities at the rollout temperature and cotangent tensors",
        "hf/step.py": "restored inputs, trajectory tensors and declared optimizer steps given to the learner",
        "hf/runtime.py": "resident learner state carried across updates",
        "binding.py": "optimizer parameter groups",
    },
}
NEUTRAL = {
    "implementation.py": "identity computation",
    "hf/assembly.py": "identity computation",
    "hf/frozen.py": "identity computation",
    "hf/tensors.py": "identity and equality checks",
    "hf/metrics.py": "time and memory measurement",
    "hf/infer.py": "request decoding; the request is echoed in the result and checked by the core",
    "hf/session.py": "request decoding; the request is echoed in the result and checked by the core",
    "hf/resident.py": "learner protocol; the consumed request is checked by the core",
    "hf/initialize.py": "initial checkpoint; its effect is fixed by the checkpoint content digests",
    "hf/state.py": "state observation and attestation",
    "invocation.py": "binding and permission protocol checked by the core",
    "registry.py": "load registry protocol checked by the core",
    "report.py": "result records checked by the core",
    "resident.py": "owner protocol checked by the core",
    "learner.py": "learner protocol checked by the core",
    "exchange.py": "learner step records and cotangent replies bound and checked by the core",
    "inputs.py": "input paths",
    "update.py": "update request decoding checked by the core",
    "cohort.py": "request field decoding checked by the core",
    "trajectory.py": "record types",
    "scalar.py": "objective scalars and cotangents recomputed bit for bit by the core",
}
IRRELEVANT = {
    INFERENCE: {},
    LEARNING: {
        "hf/operation.py": "tokenizer identity check only; the learner consumes token ids",
        "tokenization.py": "token id validation only",
        "hf/decoding.py": "generation settings, used only by inference",
    },
}


def description(root, role):
    return {"format": FORMAT, "role": role, "files": {name: file_digest(root / name) for name in ROLES[role]}}


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def current(role):
    return description(Path(__file__).resolve().parent, role)
