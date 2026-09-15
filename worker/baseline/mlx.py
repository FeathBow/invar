from contextlib import contextmanager, ExitStack
from functools import partial
import gc
import math
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_unflatten
import numpy as np

from worker.advantage import check
from worker.logical import ordered
from worker.mlx import adapter as mlx_adapter
from worker.mlx import backward as mlx_backward
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import inference as mlx_inference
from worker.mlx import learning as mlx_learning
from worker.mlx import metrics as mlx_metrics
from worker.mlx import model as mlx_model
from worker.mlx import step as mlx_step
from worker.mlx import tensors as mlx_tensors
from worker.baseline import data as product_data
from worker.record import loss
from worker import scalar
from worker.hf.step import file_digest


def update(learner, batch, *, linearize, observe_reward):
    samples = ordered(batch)
    model, optimizer = learner.model, learner.optimizer
    mlx_learning.check_optimizer(optimizer)
    count = sum(item.trajectory.behavior.size for item in samples)
    parameters = mlx_adapter.state(model)
    before = mlx_tensors.digest(parameters)
    gradients = {name: mx.zeros_like(value) for name, value in parameters.items()}
    reward_gradients = {name: mx.zeros_like(value) for name, value in parameters.items()} if observe_reward else None
    observations, reward_squares = [], []
    model.train()
    for item in samples:
        current, differentiate = linearize(model, item.trajectory, evaluate=learner.evaluate)
        mx.eval(current)
        evaluated, objective, reward = mlx_learning.observation(item, current, profile=batch.profile, total=count)
        observations.append(evaluated)
        reward_squares.append(np.square(np.asarray(reward, dtype=np.float64)).sum().item())
        contribution = differentiate(cotangent=objective)
        if contribution.keys() != parameters.keys() or any(value.dtype != mx.float32 or value.shape != parameters[name].shape
                                                           or not np.isfinite(np.asarray(value)).all() for name, value in contribution.items()):
            raise RuntimeError("Native composition produced invalid FP32 parameter gradients")
        gradients = {name: value + contribution[name] for name, value in gradients.items()}
        mx.eval(gradients)
        if observe_reward:
            contribution = differentiate(cotangent=reward)
            reward_gradients = {name: value + contribution[name] for name, value in reward_gradients.items()}
            mx.eval(reward_gradients)
        del differentiate, contribution
    norm = mlx_learning.norm(gradients)
    optimizer.update(model, tree_unflatten(list(gradients.items())))
    mx.eval(model.trainable_parameters(), optimizer.state)
    mlx_learning.check_optimizer(optimizer)
    after = mlx_tensors.digest(mlx_adapter.state(model))
    model.eval()
    summary = {"loss": scalar.number(loss(observations)), "gradient_norm": norm,
               "reward_logprob_cotangent_norm": math.sqrt(math.fsum(reward_squares)),
               "active_tokens": count, "before": before, "after": after,
               "nonzero_advantages": sum(item.advantage != 0 for item in samples),
               "parameter_vjps_per_sample": 2 if observe_reward else 1}
    if observe_reward:
        summary["reward_gradient_norm"] = mlx_learning.norm(reward_gradients)
        gradients = {**gradients, **{"reward/" + name: value for name, value in reward_gradients.items()}}
    return summary, gradients, observations


class Runtime:
    def __init__(self, runtime, options, *, settings, sampling, measure, emit, linearize):
        self.runtime = runtime
        self.options = options
        self.settings = dict(settings)
        self.sampling = sampling
        self.measure = measure
        self.emit = emit
        self.linearize = linearize
        self.learner = None
        self.reference = None
        self.optimizer = None

    @property
    def identity(self):
        return self.runtime.identity

    def generate(self, tasks):
        actual = product_data.identities(self.settings, behavior=True)
        if mlx_tensors.digest(mlx_adapter.state(self.runtime.model)) != actual["adapter"]:
            raise RuntimeError("Resident native inference policy differs from its published input")
        trajectories = self.measure("inference", lambda: mlx_inference.sample(
            self.runtime, product_data.requests(tasks), sampling=self.sampling))
        return tuple(product_data.observation(item, actual) for item in trajectories)

    def update(self, request, output):
        if (request.policy, request.learner) != (self.settings["policy"], self.settings["learner"]):
            raise ValueError("Native learner input differs from its resident successor")
        checked = check(request)
        if self.learner is None:
            paths = SimpleNamespace(checkpoint=self.options.initial, reference=self.options.reference)
            self.learner, self.reference = self.measure("restore", lambda: mlx_step.restore(self.runtime, request, paths))
            self.optimizer = request.optimizer
        elif request.optimizer != self.optimizer:
            raise ValueError("Native composition changed its resident optimizer configuration")
        admitted = mlx_step.batch(self.learner, request, self.reference, checked=checked, measure=self.measure,
                                  emit=self.emit, numerics=self.runtime.numerics)
        with self.runtime.numerics.learning(self.learner.model):
            summary, gradients, observations = self.measure("reward_update", lambda: update(
                self.learner, admitted, linearize=self.linearize, observe_reward=self.options.gradient_observation == "objective-and-reward"))
        product_data.probabilities(output, observations)
        with (output / "gradients.safetensors").open("xb") as stream:
            mx.save_safetensors(stream, gradients, metadata={"observation": "native objective gradients before AdamW"})
        saved = self.measure("checkpoint", lambda: self.save(output, summary["after"]))
        return {**summary, **saved}

    def save(self, output, policy):
        parameters = mlx_adapter.state(self.runtime.model)
        identities = {**product_data.identities(self.settings), "adapter": policy}
        saved = mlx_tensors.save_policy(output / "adapter.safetensors", parameters)
        if saved != policy:
            raise RuntimeError("Native checkpoint differs from its actual updated parameters")
        state = mlx_checkpoint.observe(self.learner.optimizer, identities=identities, parameters=parameters)
        mlx_checkpoint.save(output / "learner.pt", state)
        return {"policy": saved, "learner": file_digest(output / "learner.pt")}

    def activate(self, directory, *, settings):
        def install():
            actual = mlx_tensors.policy(directory / "adapter.safetensors", settings["policy"])
            mlx_adapter.install(self.runtime.model, actual)

        self.measure("published_activation", install)
        self.settings = dict(settings)

    def close(self):
        self.learner = None
        self.reference = None
        self.runtime = None


@contextmanager
def load(options, *, settings, emit, loader):
    configuration = mlx_model.configuration(options.configuration)
    expected = product_data.identities(settings)
    if expected != product_data.identities(settings, behavior=True):
        raise ValueError("The shared native model requires matching inference and learning materializations")
    measured = partial(mlx_metrics.measure, emit=emit)
    with ExitStack() as stack:
        runtime = loader(options.cache, scope=stack, configuration=configuration,
                         measure=measured, emit=emit, initial=(options.initial / "adapter.safetensors", expected))
        backend = Runtime(runtime, options, settings=settings, sampling=configuration.sampling(),
                          measure=measured, emit=emit, linearize=mlx_backward.linearize)
        try:
            yield backend
        finally:
            backend.close()
            del runtime
            stack.close()
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
