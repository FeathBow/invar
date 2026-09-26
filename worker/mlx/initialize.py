import argparse
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
import sys

from worker.cohort import Optimizer, identity
from worker.logical import Learner
from worker.mlx import model as mlx_model
from worker.mlx.metrics import measure
from worker.mlx.rollout import logprobs
from worker.mlx import step as mlx_step
from worker.mlx import tokenization as mlx_tokenization
from worker.resident import Transcript

INITIAL_OPTIMIZER = Optimizer(learning_rate=0.0001, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.0)


def run(options, *, loader=mlx_model.load):
    identity(options.tokenizer_digest)
    configuration = mlx_model.configuration(options.config)
    options.output.mkdir(exist_ok=False)
    output = sys.stdout
    with redirect_stdout(sys.stderr), ExitStack() as scope:
        transcript = Transcript(output)
        measured = partial(measure, emit=transcript.emit)
        loaded = loader(options.cache, scope=scope, configuration=configuration, measure=measured,
                        emit=transcript.emit, seed=options.seed, identified=True)
        mlx_tokenization.verify(loaded.tokenizer, options.tokenizer_digest)
        learner = Learner(model=loaded.model, optimizer=mlx_step.optimizer(INITIAL_OPTIMIZER), evaluate=logprobs)
        saved = measured("checkpoint", partial(mlx_step.save, loaded, learner, options.output))
        transcript.emit("initial", {name: saved[name] for name in ("policy", "learner", "tokenizer", "base", "assembly")} |
                        {"seed": options.seed, "optimizer_steps": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-digest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path)
    loaders = {"primary": mlx_model.load, "native": mlx_model.load_native}
    parser.add_argument("--numerics", choices=loaders, default="primary",
                        help="Materialize the primary numerical rule or ordinary native benchmark arithmetic")
    options = parser.parse_args()
    run(options, loader=loaders[options.numerics])


if __name__ == "__main__":
    main()
