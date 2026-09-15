import math
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from worker.advantage import check
from worker.logical import Learner, ordered
from worker.hf import learning
from worker.hf.objective import Tokens
from worker.hf.probability import checked, cotangents, loss
from worker.hf import probe
from worker.baseline import data as product_data
from worker import scalar
from worker.hf import step


def update(learner, batch, *, observe_reward):
    samples = ordered(batch)
    model, optimizer = learner.model, learner.optimizer
    learning.check_optimizer(optimizer)
    count = sum(item.trajectory.behavior.numel() for item in samples)
    before = probe.digest(probe.adapter_state(model))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    observations, reward_squares = [], []
    trainable = learning.parameters(model)
    reward_gradients = tuple(torch.zeros_like(value) for value in trainable) if observe_reward else None
    for item in samples:
        current = learner.evaluate(model, item.trajectory)
        tokens = Tokens(current=current, proximal=item.proximal.to(current.device),
                        reference=item.reference.to(current.device), behavior=item.trajectory.behavior.to(current.device),
                        advantage=torch.full_like(current, item.advantage), active=torch.ones_like(current, dtype=torch.bool))
        observed = checked(item.trajectory.request.sample, tokens, advantage=item.advantage,
                           count=item.trajectory.tokens.shape[-1] - item.trajectory.prompt_length)
        evaluated, objective, reward = cotangents(observed, batch.profile, total=count, device=current.device)
        observations.append(evaluated)
        reward_squares.append(reward.double().square().sum().item())
        if observe_reward:
            contribution = torch.autograd.grad(current, trainable, grad_outputs=reward, retain_graph=True)
            reward_gradients = tuple(total + value for total, value in zip(reward_gradients, contribution, strict=True))
        current.backward(objective)
    named = tuple((name, value) for name, value in model.named_parameters() if value.requires_grad)
    norm = learning.gradient_norm(tuple(value.grad for _, value in named))
    gradients = {name: value.grad.detach().cpu().contiguous().clone() for name, value in named}
    optimizer.step()
    learning.check_optimizer(optimizer)
    if any(not value.isfinite().all() for _, value in named):
        raise RuntimeError("Native learner produced nonfinite parameters")
    model.eval()
    summary = {"loss": scalar.number(loss(observations)), "gradient_norm": norm,
               "reward_logprob_cotangent_norm": math.sqrt(math.fsum(reward_squares)),
               "active_tokens": count, "before": before, "after": probe.digest(probe.adapter_state(model)),
               "nonzero_advantages": sum(item.advantage != 0 for item in samples),
               "parameter_vjps_per_sample": 2 if observe_reward else 1}
    if observe_reward:
        summary["reward_gradient_norm"] = learning.gradient_norm(reward_gradients)
        gradients.update({"reward/" + name: value.detach().cpu().contiguous().clone()
                          for (name, _), value in zip(named, reward_gradients, strict=True)})
    return summary, gradients, observations


class Runtime:
    def __init__(self, model, tokenizer, options, *, settings, measure, emit, evaluate):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.settings = dict(settings)
        self.measure = measure
        self.emit = emit
        self.evaluate = evaluate
        self.learner = None
        self.reference = None
        self.optimizer = None

    def update(self, request, output):
        if (request.policy, request.learner) != (self.settings["policy"], self.settings["learner"]):
            raise ValueError("Native learner input differs from its resident successor")
        checked = check(request)
        if self.learner is None:
            paths = SimpleNamespace(checkpoint=self.options.initial, reference=self.options.reference)
            optimizer, self.reference, _ = self.measure("restore", lambda: step.restore_inputs(
                self.model, request, paths, tokenizer=self.tokenizer))
            self.learner = Learner(model=self.model, optimizer=optimizer, evaluate=self.evaluate)
            self.optimizer = request.optimizer
        elif request.optimizer != self.optimizer:
            raise ValueError("Native composition changed its resident optimizer configuration")
        admitted = step.batch(self.model, request, self.reference, checked=checked, measure=self.measure,
                              evaluate=self.evaluate, emit=self.emit)
        summary, gradients, observations = self.measure("reward_update", lambda: update(
            self.learner, admitted, observe_reward=self.options.gradient_observation == "objective-and-reward"))
        product_data.probabilities(output, observations)
        save_file(gradients, output / "gradients.safetensors", metadata={"observation": "native objective gradients before AdamW"})
        saved = self.measure("checkpoint", lambda: step.checkpoint_update(
            self.learner, request, output, tokenizer=self.tokenizer, summary=summary))
        return {**summary, "policy": probe.digest(saved), "learner": step.file_digest(output / "learner.pt")}

    def activate(self, directory, *, settings):
        if probe.digest(probe.adapter_state(self.model)) != settings["policy"]:
            raise RuntimeError("Native learner state differs from its published successor")
        self.settings = dict(settings)

    def close(self):
        self.learner = None
        self.reference = None
        self.model = None
