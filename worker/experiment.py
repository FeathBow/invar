import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from learning import Batch, Learner, Sample, parameters, probabilities, update
from objective import Profile, Reward, advantages
from probe import (ADAM_BETAS, ADAM_EPSILON, CPU_THREADS, DEFAULT_SEED, LEARNING_RATE,
                   MODEL, REVISION, adapter_state, checkpoint, digest, load_model, measure,
                   report, restore)
from rollout import Request, generate, logprobs, reward

DATA_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TRAIN_DIGEST = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"
GROUP_SIZE = 4
GENERATIONS = 2
TOKEN_LIMIT = 256
TEMPERATURE = 0.8
EPSILON = 0.2
PENALTY = 0.04
DELTA = 1e-4
DATASET_SIZE = 7473
INSTRUCTION = "\nReason briefly. End your answer with a final line in the form #### <number>."


def workload(path, indices):
    if hashlib.sha256(path.read_bytes()).hexdigest() != TRAIN_DIGEST:
        raise ValueError("Training data does not match the frozen GSM8K main artifact")
    rows = parquet.read_table(path, columns=["question", "answer"]).to_pylist()
    if len(rows) != DATASET_SIZE:
        raise ValueError("Frozen training dataset is incomplete")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Training rows must be nonempty and distinct")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError("Training row is outside the frozen dataset")
    return tuple((index, rows[index]) for index in indices)


def collect(model, tokenizer, context):
    generation, rows = context
    trajectories, rewards = [], []
    for row, item in rows:
        group = f"train/{row}/generation/{generation}"
        expected = "#### " + item["answer"].rsplit("#### ", 1)[-1].strip().replace(",", "")
        for position in range(GROUP_SIZE):
            request = Request(sample=f"{group}/sample/{position}", group=group,
                              prompt=item["question"] + INSTRUCTION,
                              seed=DEFAULT_SEED + (generation * DATASET_SIZE + row) * GROUP_SIZE + position,
                              limit=TOKEN_LIMIT, temperature=TEMPERATURE)
            trajectory = generate(model, tokenizer, request)
            score = reward(trajectory.text, expected, trajectory.truncated)
            trajectories.append(trajectory)
            rewards.append(Reward(sample=request.sample, group=group, value=score))
            report("rollout", {"sample": request.sample, "seed": request.seed,
                               "tokens": trajectory.tokens.tolist(), "prompt_length": trajectory.prompt_length,
                               "behavior": trajectory.behavior.tolist(), "text": trajectory.text,
                               "truncated": trajectory.truncated, "reward": score})
    return tuple(trajectories), tuple(rewards)


def learning_cycle(model, optimizer, context):
    generation, tokenizer, rows, reference, output = context
    order = tuple(f"train/{row}/generation/{generation}/sample/{position}"
                  for row, _ in rows for position in range(GROUP_SIZE))
    trajectories, rewards = measure("rollout_group", lambda: collect(model, tokenizer, (generation, rows)))
    proximal, fixed = measure("probability_roles", lambda: probabilities(model, trajectories, reference, evaluate=logprobs))
    values = dict(advantages(rewards, DELTA))
    samples = tuple(Sample(trajectory=item, proximal=old, reference=ref, advantage=values[item.request.sample])
                    for item, old, ref in zip(trajectories, proximal, fixed, strict=True))
    for item in samples:
        report("roles", {"sample": item.trajectory.request.sample, "proximal": item.proximal.tolist(),
                         "reference": item.reference.tolist(), "advantage": item.advantage})
    learner = Learner(model=model, optimizer=optimizer, evaluate=logprobs)
    result = measure("reward_update", lambda: update(learner, Batch(
        samples=samples, order=order, profile=Profile(epsilon=EPSILON, penalty=PENALTY))))
    report("learning", {"generation": generation, **result.summary})
    state = output / f"generation-{generation}"
    state.mkdir()
    saved = checkpoint(model, optimizer, state, tokenizer=tokenizer, expected=None)
    restore(model, optimizer, state, tokenizer=tokenizer)
    report("checkpoint", {"generation": generation, "adapter": digest(saved),
                          "reloaded": digest(adapter_state(model)), "atomic_publication": False})
    return result.summary["reward_gradient_norm"] > 0 and result.summary["before"] != result.summary["after"]


def arguments():
    parser = argparse.ArgumentParser(description="Reward-driven tensor-worker experiment; not an Invar release")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    return parser.parse_args()


def main():
    options = arguments()
    rows = workload(options.data, options.rows)
    options.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    report("experiment", {"dataset": "openai/gsm8k", "revision": DATA_REVISION,
                          "split": "main/train", "rows": options.rows,
                          "seed_rule": "base + (generation * dataset_size + row) * group_size + sample",
                          "base_seed": DEFAULT_SEED, "dataset_size": DATASET_SIZE,
                          "reduction_order": "declared row sequence, then sample position",
                          "group_size": GROUP_SIZE, "generations": GENERATIONS,
                          "token_limit": TOKEN_LIMIT, "temperature": TEMPERATURE,
                          "epsilon": EPSILON, "penalty": PENALTY, "delta": DELTA,
                          "scope": "reward-driven execution, checkpoint and reload; no semantic driver certification"})
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=options.cache, local_files_only=True)
    model = measure("load", lambda: load_model(path))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    reference = adapter_state(model)
    optimizer = torch.optim.AdamW(parameters(model), lr=LEARNING_RATE, betas=ADAM_BETAS,
                                  eps=ADAM_EPSILON, weight_decay=0.0, foreach=False, fused=False)
    results = [learning_cycle(model, optimizer, (generation, tokenizer, rows, reference, options.output))
               for generation in range(GENERATIONS)]
    report("result", {"reward_driven_updates": results, "v0_1_complete": False})
    if not all(results):
        raise RuntimeError("The frozen workload did not produce two nonzero reward-driven updates")


if __name__ == "__main__":
    main()
