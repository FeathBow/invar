from contextlib import contextmanager
from functools import partial
import gc
from types import SimpleNamespace

from worker.hf.metrics import measure
from worker.baseline.process import open_owner
from worker.hf.rollout import logprobs
from worker.baseline import data as product_data
from worker.hf import step
from worker.baseline import learner as torch_product
import torch


class Runtime:
    def __init__(self, inference, learner, *, settings, identity):
        self.inference = inference
        self.learner = learner
        self.settings = dict(settings)
        self.identity = identity

    def generate(self, tasks):
        return self.inference.exchange("generate", {"tasks": tasks})["results"]

    def update(self, request, output):
        return self.learner.update(request, output)

    def activate(self, directory, *, settings):
        observed = self.inference.exchange("activate", {"adapter": str(directory / "adapter.safetensors"), "policy": settings["policy"]})
        if observed["identities"] != product_data.identities(settings, behavior=True):
            raise RuntimeError("Native inference did not activate the published successor")
        self.learner.activate(directory, settings=settings)
        self.settings = dict(settings)


@contextmanager
def load(options, *, settings, emit):
    timed = partial(measure, emit=emit)
    with open_owner(options, settings=settings) as (inference, ready):
        actual = ready["identities"]
        if actual != product_data.identities(settings, behavior=True):
            raise ValueError("Native product engine differs from the declared inference materialization")
        model, tokenizer, _ = step.load_with(options, SimpleNamespace(tokenizer=settings["tokenizer-digest"]), measure=timed, emit=emit)
        learner = torch_product.Runtime(model, tokenizer, options, settings=settings, measure=timed, emit=emit, evaluate=logprobs)
        try:
            yield Runtime(inference, learner, settings=settings, identity=(ready["model"], ready["revision"]))
        finally:
            learner.close()
            del model
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
