import argparse
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

from update import consumed, decode, snapshot
from invocation import approve


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def optimizer_options(settings):
    return {"lr": settings.learning_rate, "betas": settings.betas, "eps": settings.epsilon,
            "weight_decay": settings.weight_decay, "foreach": False, "fused": False,
            "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False}


def restore_inputs(model, request, options, *, tokenizer):
    import torch
    from learning import check_optimizer, parameters
    from policy import read_adapter, validate_schema
    from probe import adapter_state, digest, restore_state
    from tokenization import validate
    from operation import verify

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
    group = optimizer.param_groups[0]
    loaded = {"policy": policy_digest, "learner": learner_digest, "reference": digest(reference),
              "tokenizer": tokenizer_digest, "base": training["base"], "assembly": training["assembly"],
              "optimizer": {"learning_rate": group["lr"], "betas": group["betas"],
                            "epsilon": group["eps"], "weight_decay": group["weight_decay"]}}
    return optimizer, reference, loaded


def trajectory(item):
    import torch
    from rollout import Request, Trajectory

    numerical = Request(sample=item.sample, group=item.group, prompt=item.prompt,
                        seed=item.seed, limit=item.limit, temperature=item.temperature)
    encoded = bytearray(struct.pack(f"={len(item.behavior_bits)}I", *item.behavior_bits))
    behavior = torch.frombuffer(encoded, dtype=torch.float32).clone()
    return Trajectory(request=numerical, tokens=torch.tensor([item.tokens], dtype=torch.long),
                      prompt_length=item.prompt_length, behavior=behavior, text=item.text,
                      truncated=item.truncated)


def batch(model, request, reference, *, measure, evaluate):
    from learning import Batch, Sample, probabilities
    from objective import Profile, Reward, advantages
    from probe import report

    trajectories = tuple(trajectory(item) for item in request.samples)
    rewards = tuple(Reward(sample=item.sample, group=item.group, value=item.reward) for item in request.samples)
    proximal, fixed = measure("probability_roles", lambda: probabilities(model, trajectories, reference, evaluate=evaluate))
    normalized = dict(advantages(rewards, request.delta))
    samples = tuple(Sample(trajectory=item, proximal=old, reference=ref, advantage=normalized[item.request.sample])
                    for item, old, ref in zip(trajectories, proximal, fixed, strict=True))
    for item in samples:
        report("roles", {"sample": item.trajectory.request.sample,
                         "proximal_policy": request.policy, "reference_policy": request.reference,
                         "proximal": item.proximal.tolist(), "reference": item.reference.tolist(),
                         "advantage": item.advantage})
    return (Batch(samples=samples, order=request.order,
                  profile=Profile(epsilon=request.epsilon, penalty=request.penalty)), rewards)


def checkpoint_update(learner, request, output, *, tokenizer, summary):
    from probe import CheckpointIdentity, checkpoint

    if summary["before"] != request.policy:
        raise RuntimeError("Update input differs from the declared policy")
    expected = CheckpointIdentity(adapter=summary["after"], base=request.base,
                                  assembly=request.assembly, tokenizer=request.tokenizer)
    return checkpoint(learner.model, learner.optimizer, output, tokenizer=tokenizer, expected=expected)


def load(options, request):
    import torch
    from huggingface_hub import snapshot_download
    from operation import load as load_tokenizer, verify
    from probe import CPU_THREADS, DEFAULT_SEED, MODEL, REVISION, load_model, measure

    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=options.cache, local_files_only=True)
    tokenizer = load_tokenizer(path)
    verify(tokenizer, request.tokenizer)
    model = measure("load", lambda: load_model(path))
    return model, tokenizer, (MODEL, REVISION)


def run(call, options, *, loader, measure, permission, evaluate):
    from learning import Learner, update
    from registry import invocation, learning
    from safetensors.torch import save_file
    from probability import save as save_probabilities
    from probe import digest, report

    request = call.request
    bound = call.invocation.binding()
    options.output.mkdir(exist_ok=False)
    model, tokenizer, identity = loader(options, request)
    optimizer, reference, loaded = restore_inputs(model, request, options, tokenizer=tokenizer)
    report("loaded_learner", {"binding": bound, "state": loaded, "load": invocation(call.load),
                              "image": learning(loaded), "model": identity[0], "revision": identity[1]})
    admitted, rewards = batch(model, request, reference, measure=measure, evaluate=evaluate)
    actual = consumed(request, batch=admitted, rewards=rewards, loaded=loaded)
    report("consumed", {"binding": bound, "program": call.invocation.program, "request": actual,
                         "load": invocation(call.load)})
    permission(call.invocation)
    learner = Learner(model=model, optimizer=optimizer, evaluate=evaluate)
    result = measure("reward_update", lambda: update(learner, admitted))
    probability_digest = save_probabilities(options.output / "probabilities.json", result.probabilities,
                                            invocation={"binding": bound, "program": call.invocation.program},
                                            request=actual)
    gradients = options.output / "gradients.safetensors"
    save_file(result.gradients, gradients, metadata={"binding": json.dumps(bound, sort_keys=True),
                                                   "program": call.invocation.program,
                                                   "policy": request.policy,
                                                   "observation": "objective and reward gradients before AdamW"})
    saved = checkpoint_update(learner, request, options.output, tokenizer=tokenizer, summary=result.summary)
    report("result", {"binding": bound, "request": actual, "update": result.summary,
                      "gradients": file_digest(gradients),
                      "probabilities": probability_digest,
                      "adapter": digest(saved), "learner": file_digest(options.output / "learner.pt"),
                      "storage": "staged; not published"})


def arguments():
    parser = argparse.ArgumentParser(description="Execute one declared GRPO update and stage its checkpoint")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    from session import unique
    from probe import measure
    from rollout import logprobs

    options = arguments()
    call = decode(json.loads(sys.stdin.readline(), object_pairs_hook=unique))
    run(call, options, loader=load, measure=measure, permission=approve, evaluate=logprobs)


if __name__ == "__main__":
    main()
