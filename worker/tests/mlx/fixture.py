import unittest

try:
    import mlx  # noqa: F401
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mlx.core as mx
from mlx_lm.generate import wired_limit

from worker.mlx import infer as mlx_infer
from worker.mlx import cohort as mlx_inference_resident
from worker.mlx import initialize as mlx_initialize
from worker.mlx import model as mlx_model
from worker.mlx import resident as mlx_resident
from worker.tests.mlx.rollout import model, tokenizer
from worker.implementation import INFERENCE

IDENTITY = ("invar-native-hybrid-test", "fixture-v1")


def load(cache, *, scope, configuration, measure, emit, seed=17, initial=None, identified=False, learnable=False):
    mx.random.seed(seed)
    previous = mx.set_cache_limit(configuration.cache_bytes)
    scope.callback(mx.set_cache_limit, previous)

    def prepare():
        emit("loading", {"model": IDENTITY[0], "revision": IDENTITY[1]})
        numerical, config = model(uniform_head=not learnable)
        configuration.profile().install(numerical)
        numerical.eval()
        scope.enter_context(wired_limit(numerical))
        loaded = mlx_model.Loaded(model=numerical, tokenizer=tokenizer(), config=config, identity=IDENTITY,
                                  numerics=configuration.profile())
        if initial is not None:
            mlx_model.activate(loaded, initial[0], expected=initial[1])
        reported = {"model": IDENTITY[0], "revision": IDENTITY[1],
                    "native_test": {"layers": len(numerical.layers), "uniform_head": not learnable,
                                    "batch_size": configuration.batch_size}}
        if identified:
            reported["inference"] = {name: value for name, value in mlx_model.identities(loaded, INFERENCE).items()
                                     if name in ("base", "assembly")}
        emit("profile", reported)
        with (cache / "loads.txt").open("a") as output:
            output.write(str(os.getpid()) + "\n")
        return loaded

    return measure("load", prepare)


def main(loader=load):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tokenizer-digest")
    parser.add_argument("--digest")
    parser.add_argument("--base-digest")
    parser.add_argument("--assembly-digest")
    parser.add_argument("--prompt")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--session", type=int)
    parser.add_argument("--shared", action="store_true")
    options = parser.parse_args()
    if options.shared:
        mlx_resident.run(options, loader=loader)
    elif options.session is not None:
        mlx_inference_resident.run(options, loader=loader)
    elif options.output is not None:
        mlx_initialize.run(options, loader=loader)
    else:
        mlx_infer.run(options, loader=loader)


if __name__ == "__main__":
    main()
