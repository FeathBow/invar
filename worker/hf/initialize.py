import argparse
from pathlib import Path

from worker.cohort import identity


def arguments():
    parser = argparse.ArgumentParser(description="Materialize a fresh bound initial LoRA learner without updating it")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-digest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    options = parser.parse_args()
    identity(options.tokenizer_digest)
    if options.output.exists():
        parser.error("--output must not already exist")
    return options


def main():
    options = arguments()
    import torch
    from huggingface_hub import snapshot_download

    from worker.hf.learning import parameters
    from worker.hf.probe import (ADAM_BETAS, ADAM_EPSILON, CPU_THREADS, LEARNING_RATE, MODEL,
                       REVISION, checkpoint, digest, load_model, measure, report)
    from worker.hf.step import file_digest
    from worker.hf.operation import load, verify

    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=options.cache, local_files_only=True)
    tokenizer = load(path)
    observed = verify(tokenizer, options.tokenizer_digest)
    model = measure("load", lambda: load_model(path))
    optimizer = torch.optim.AdamW(parameters(model), lr=LEARNING_RATE, betas=ADAM_BETAS,
                                  eps=ADAM_EPSILON, weight_decay=0.0, foreach=False, fused=False)
    options.output.mkdir(exist_ok=False)
    saved = measure("checkpoint", lambda: checkpoint(model, optimizer, options.output, tokenizer=tokenizer, expected=None))
    learner = torch.load(options.output / "learner.pt", weights_only=True)
    report("initial", {"policy": digest(saved), "learner": file_digest(options.output / "learner.pt"),
                       "tokenizer": observed, "base": learner["base"], "assembly": learner["assembly"],
                       "seed": options.seed,
                       "optimizer_steps": 0, "scope": "materialized initial learner; not publication or qualification"})


if __name__ == "__main__":
    main()
