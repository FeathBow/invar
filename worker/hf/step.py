import argparse
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

from worker.update import consumed, decode, snapshot
from worker.invocation import approve


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def optimizer_options(settings):
    return {"lr": settings.learning_rate, "betas": settings.betas, "eps": settings.epsilon,
            "weight_decay": settings.weight_decay, "foreach": False, "fused": False,
            "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False}


def restore_inputs(model, request, options, *, tokenizer):
    import torch
    from worker.hf.learning import check_optimizer, parameters
    from worker.hf.policy import read_adapter, validate_schema
    from worker.hf.model import adapter_state
    from worker.hf.tensors import digest
    from worker.hf.checkpoint import restore_state
    from worker.tokenization import validate
    from worker.hf.operation import verify

    tokenizer_digest = verify(tokenizer, request.tokenizer)
    validate(tokenizer, request.samples)
    policy = read_adapter(options.checkpoint / "adapter.safetensors", request.policy)
    reference = read_adapter(options.reference, request.reference)
    schema = adapter_state(model)
    validate_schema(schema, policy)
    validate_schema(schema, reference)
    learner_digest, encoded = snapshot(options.checkpoint / "learner.pt")
    if learner_digest != request.learner:
        raise ValueError("Learner file differs from the requested checkpoint bytes")
    expected = optimizer_options(request.optimizer)
    optimizer = torch.optim.AdamW(parameters(model), **expected)
    training = torch.load(io.BytesIO(encoded), weights_only=True)
    if training["base"] != request.base or training["assembly"] != request.assembly:
        raise ValueError("Checkpoint model materialization differs from the requested update input")
    restore_state(model, optimizer, (policy, training), tokenizer=tokenizer)
    policy_digest = digest(adapter_state(model))
    if policy_digest != request.policy:
        raise RuntimeError("Loaded policy differs from the declared update input")
    for group in optimizer.param_groups:
        if any(group[name] != value for name, value in expected.items()):
            raise ValueError("Restored optimizer differs from the declared update specification")
    check_optimizer(optimizer)
    loaded = loaded_inputs(optimizer, {"policy": policy_digest, "learner": learner_digest, "reference": digest(reference),
                                       "tokenizer": tokenizer_digest, "base": training["base"], "assembly": training["assembly"]})
    return optimizer, reference, loaded


def loaded_inputs(optimizer, identities):
    group = optimizer.param_groups[0]
    return {**identities, "optimizer": {"learning_rate": group["lr"], "betas": group["betas"],
                                       "epsilon": group["eps"], "weight_decay": group["weight_decay"]}}


def trajectory(item):
    import torch
    from worker.trajectory import Request, Trajectory

    numerical = Request(sample=item.sample, group=item.group, prompt=item.prompt,
                        seed=item.seed, limit=item.limit, temperature=item.temperature)
    encoded = bytearray(struct.pack(f"={len(item.behavior_bits)}I", *item.behavior_bits))
    behavior = torch.frombuffer(encoded, dtype=torch.float32).clone()
    return Trajectory(request=numerical, tokens=torch.tensor([item.tokens], dtype=torch.long),
                      prompt_length=item.prompt_length, behavior=behavior, text=item.text,
                      truncated=item.truncated)


def batch(request, *, checked, measure, emit):
    from worker.hf.learning import Batch, Sample, probabilities
    from worker.hf.objective import Profile

    trajectories = tuple(trajectory(item) for item in request.samples)
    scores = tuple(item.reference_bits for item in request.samples)
    proximal, fixed = measure("probability_roles", lambda: probabilities(trajectories, scores))
    normalized = dict(checked.values)
    samples = tuple(Sample(trajectory=item, proximal=old, reference=ref, advantage=normalized[item.request.sample])
                    for item, old, ref in zip(trajectories, proximal, fixed, strict=True))
    for item in samples:
        emit("roles", {"sample": item.trajectory.request.sample,
                         "proximal_policy": request.policy, "reference_policy": request.reference,
                         "proximal": item.proximal.tolist(), "reference": item.reference.tolist(),
                         "advantage": item.advantage})
    return Batch(samples=samples, order=request.order,
                 profile=Profile(epsilon=request.epsilon, penalty=request.penalty))


def checkpoint_update(learner, request, output, *, tokenizer, summary):
    from worker.hf.checkpoint import CheckpointIdentity, checkpoint

    if summary["before"] != request.policy:
        raise RuntimeError("Update input differs from the declared policy")
    expected = CheckpointIdentity(adapter=summary["after"], base=request.base,
                                  assembly=request.assembly, tokenizer=request.tokenizer)
    return checkpoint(learner.model, learner.optimizer, output, tokenizer=tokenizer, expected=expected)


def load(options, request):
    from worker.hf.metrics import measure, report

    return load_with(options, request, measure=measure, emit=report)


def load_with(options, request, *, measure, emit):
    import torch
    from huggingface_hub import snapshot_download
    from worker.hf.operation import load as load_tokenizer, verify
    from worker.hf.model import CPU_THREADS, DEFAULT_SEED, MODEL, REVISION, load_model
    from worker.implementation import LEARNING

    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=options.cache, local_files_only=True)
    tokenizer = load_tokenizer(path)
    verify(tokenizer, request.tokenizer)
    model = measure("load", lambda: load_model(path, role=LEARNING, emit=emit))
    return model, tokenizer, (MODEL, REVISION)


def run(call, options, *, loader, measure, permission, evaluate):
    from worker.advantage import check
    from worker.hf.learning import Learner
    from worker.hf.metrics import report

    checked = check(call.request)
    options.output.mkdir(exist_ok=False)
    model, tokenizer, identity = loader(options, call.request)
    optimizer, _, loaded = restore_inputs(model, call.request, options, tokenizer=tokenizer)
    admitted, actual = consume(call, loaded=loaded, identity=identity, checked=checked, measure=measure, emit=report)
    permission(call.invocation)
    learner = Learner(model=model, optimizer=optimizer, evaluate=evaluate)
    report("result", execute(learner, call, options.output, batch=admitted, actual=actual,
                             tokenizer=tokenizer, measure=measure))


def consume(call, *, loaded, identity, checked, measure, emit):
    from worker.registry import invocation, learning
    request = call.request
    bound = call.invocation.binding()
    emit("loaded_learner", {"binding": bound, "state": loaded, "load": invocation(call.load),
                            "image": learning(loaded), "model": identity[0], "revision": identity[1]})
    admitted = batch(request, checked=checked, measure=measure, emit=emit)
    actual = consumed(request, batch=admitted, rewards=checked.rewards, loaded=loaded)
    emit("consumed", {"binding": bound, "program": call.invocation.program, "request": actual,
                       "load": invocation(call.load)})
    return admitted, actual


def execute(learner, call, output, *, batch, actual, tokenizer, measure):
    from worker.hf.learning import update
    from safetensors.torch import save_file
    from worker.hf.probability import save as save_probabilities
    from worker.hf.tensors import digest

    request = call.request
    bound = call.invocation.binding()
    result = measure("reward_update", lambda: update(learner, batch))
    probability_digest = save_probabilities(output / "probabilities.json", result.probabilities,
                                            invocation={"binding": bound, "program": call.invocation.program},
                                            request=actual)
    gradients = output / "gradients.safetensors"
    save_file(result.gradients, gradients, metadata={"binding": json.dumps(bound, sort_keys=True),
                                                   "program": call.invocation.program,
                                                   "policy": request.policy,
                                                   "observation": "objective and reward gradients before AdamW"})
    saved = checkpoint_update(learner, request, output, tokenizer=tokenizer, summary=result.summary)
    return {"binding": bound, "request": actual, "update": result.summary,
            "gradients": file_digest(gradients), "probabilities": probability_digest,
            "adapter": digest(saved), "learner": file_digest(output / "learner.pt"),
            "storage": "staged; not published"}


def arguments():
    parser = argparse.ArgumentParser(description="Execute one declared GRPO update and stage its checkpoint")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    from worker.cohort import unique
    from worker.hf.metrics import measure
    from worker.hf.probability import logprobs

    options = arguments()
    call = decode(json.loads(sys.stdin.readline(), object_pairs_hook=unique))
    run(call, options, loader=load, measure=measure, permission=approve, evaluate=logprobs)


if __name__ == "__main__":
    main()
