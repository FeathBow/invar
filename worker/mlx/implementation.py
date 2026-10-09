from pathlib import Path

from worker.implementation import INFERENCE, LEARNING, file_digest

FORMAT = "invar-mlx-implementation-v1"
SHARED = {
    "mlx/model.py": "model loading, quantization and numerical profile installation",
    "mlx/numerics.py": "numerical profile installation and observation",
    "mlx/projection.py": "row-blocked projection and LoRA column padding",
    "mlx/attention.py": "query-blocked attention",
    "mlx/recurrence.py": "segmented recurrence bound into the native gated delta call",
    "mlx/cache.py": "KV and recurrent caches",
    "mlx/adapter.py": "adapter installation and LoRA layers",
    "mlx/temperature.py": "tempered distribution shared by the sampler and the learner",
    "mlx/words.py": "FP32 word representation",
}
ROLES = {
    INFERENCE: {
        **SHARED,
        "mlx/inference.py": "execution order, sampler randomness and reference scoring",
        "mlx/rollout.py": "sampling loop and batch generation",
        "mlx/crossscore.py": "forced-path scoring",
        "mlx/distribution.py": "full-vocabulary probe capture",
        "mlx/score.py": "scoring execution",
        "mlx/tokenization.py": "prompt tokens",
        "hf/operation.py": "tokenizer loading: chat template and special tokens",
        "tokenization.py": "response text decoding",
        "scoring.py": "prescribed token paths",
        "distribution.py": "probe snapshot values",
        "probestore.py": "probe word storage",
        "probeoutput.py": "probe word encoding",
    },
    LEARNING: {
        **SHARED,
        "mlx/learning.py": "learner graph linearization, per-step gradient accumulation and AdamW steps in the declared order",
        "mlx/training.py": "learner log probabilities, the learning projection and its description",
        "mlx/backward.py": "layerwise vector-Jacobian products",
        "mlx/recurrentvjp.py": "recurrent vector-Jacobian products",
        "mlx/probability.py": "cotangent tensors",
        "mlx/step.py": "restored inputs, trajectory tensors and declared optimizer steps given to the learner",
        "mlx/checkpoint.py": "optimizer and parameter state saved and restored",
        "mlx/state.py": "random and optimizer state restored between updates",
        "mlx/learner.py": "resident learner state carried across updates",
        "logical.py": "samples and trajectories of each declared optimizer step",
    },
}
NEUTRAL = {
    "implementation.py": "identity computation",
    "mlx/implementation.py": "identity computation",
    "mlx/tensors.py": "identity and equality checks",
    "mlx/metrics.py": "time and memory measurement",
    "mlx/infer.py": "process entry and argument parsing",
    "mlx/batch.py": "process entry and argument parsing",
    "mlx/scoring.py": "process entry and argument parsing",
    "mlx/cohort.py": "process entry and group dispatch",
    "mlx/initialize.py": "initial checkpoint; its effect is fixed by the checkpoint content digests",
    "hf/infer.py": "request decoding; the request is echoed in the result and checked by the core",
    "session.py": "request decoding; the request is echoed in the result and checked by the core",
    "hf/metrics.py": "time and memory measurement",
    "hf/tensors.py": "identity and equality checks",
    "hf/frozen.py": "identity computation",
    "batch.py": "batch protocol checked by the core",
    "cohort.py": "request field decoding checked by the core",
    "core.py": "core invocation protocol",
    "dispatch.py": "resident dispatch protocol checked by the core",
    "invocation.py": "binding and permission protocol checked by the core",
    "registry.py": "load registry protocol checked by the core",
    "report.py": "result records checked by the core",
    "resident.py": "owner protocol checked by the core",
    "update.py": "update request decoding checked by the core",
    "trajectory.py": "record types",
    "roles.py": "probability role sources, recorded by content in the learning assembly description",
    "exchange.py": "learner step records and cotangent replies bound and checked by the core",
    "float32.py": "FP32 word conversion for probe values, behavior words and optimizer settings",
    "record.py": "probability records the core checks against the step exchange",
}
UNREACHED = {
    name: "Hugging Face inference path, imported only inside the Hugging Face entry functions of hf/infer and hf/operation"
    for name in ("hf/inference.py", "hf/model.py", "hf/rollout.py", "hf/decoding.py", "hf/backend.py",
                 "hf/assembly.py", "hf/policy.py", "hf/score.py")
}
IRRELEVANT = {
    INFERENCE: {
        "mlx/training.py": "learner log probabilities and the learning description, used only for the learning role",
    },
    LEARNING: {
        "mlx/inference.py": "inference execution, used by the learner process only for rollouts",
        "mlx/rollout.py": "sampling loop and batch generation; the learner uses mlx/temperature.py",
        "mlx/crossscore.py": "forced-path scoring, inference only",
        "mlx/distribution.py": "probe capture, inference only",
        "mlx/tokenization.py": "tokenizer identity check only; the learner consumes token ids",
        "hf/operation.py": "tokenizer loading, inference only",
        "tokenization.py": "text decoding, inference only",
        "scoring.py": "prescribed paths, inference only",
        "distribution.py": "probe values, inference only",
        "probestore.py": "probe storage, inference only",
    },
}


def description(root, role):
    return {"format": FORMAT, "role": role, "files": {name: file_digest(root / name) for name in ROLES[role]}}


def current(role):
    return description(Path(__file__).resolve().parents[1], role)
