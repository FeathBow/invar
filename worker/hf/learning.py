
import torch

from worker.hf.objective import Tokens
from worker.hf.probe import adapter_state, digest
from worker.hf.probability import checked, cotangents, loss, tensor
from worker.logical import Batch, Learner, Result, Sample, ordered
from worker.scalar import number


def probabilities(trajectories, scores):
    proximal = tuple(item.behavior for item in trajectories)
    if not any(scores):
        return proximal, proximal
    return proximal, tuple(tensor(words, device="cpu") for words in scores)


def parameters(model):
    return tuple(value for value in model.parameters() if value.requires_grad)


def gradient_norm(gradients):
    if not gradients or any(value is None or not value.isfinite().all() for value in gradients):
        raise RuntimeError("Missing or non-finite learner gradients")
    return torch.stack([value.double().square().sum() for value in gradients]).sum().sqrt().item()


def check_optimizer(optimizer):
    if any(not slot.isfinite().all() for slots in optimizer.state.values() for slot in slots.values()):
        raise RuntimeError("Non-finite optimizer state")


def capture_gradients(model, reward_gradients):
    named = tuple((name, value) for name, value in model.named_parameters() if value.requires_grad)
    recorded = {}
    for (name, parameter), reward in zip(named, reward_gradients, strict=True):
        recorded[f"objective/{name}"] = parameter.grad.detach().cpu().contiguous().clone()
        recorded[f"reward/{name}"] = reward.detach().cpu().contiguous().clone()
    return recorded


def parameter_vjps(current, trainable, *, objective, reward):
    # A fixed two-role batch reuses the same forward graph for both adjoints.
    gradients = torch.autograd.grad(current, trainable,
                                    grad_outputs=torch.stack((reward, objective)), is_grads_batched=True)
    reward_values, objective_values = zip(*(value.unbind() for value in gradients), strict=True)
    return objective_values, reward_values


def accumulate_objective(trainable, contribution):
    for parameter, gradient in zip(trainable, contribution, strict=True):
        parameter.grad = gradient.clone() if parameter.grad is None else parameter.grad + gradient


def update(learner, batch):
    samples = ordered(batch)
    model, optimizer = learner.model, learner.optimizer
    check_optimizer(optimizer)
    count = sum(item.trajectory.behavior.numel() for item in samples)
    before = digest(adapter_state(model))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    observations = []
    trainable = parameters(model)
    reward_gradients = tuple(torch.zeros_like(value) for value in trainable)
    for item in samples:
        linearized = learner.evaluate(model, item.trajectory)
        if linearized.dtype != torch.float32 or linearized.shape != item.trajectory.behavior.shape:
            raise ValueError("Learner graph values must be FP32 response vectors")
        current = item.proximal.to(linearized.device)
        tokens = Tokens(current=current, proximal=current,
                        behavior=item.trajectory.behavior.to(current.device),
                        reference=item.reference.to(current.device),
                        advantage=torch.full_like(current, item.advantage),
                        active=torch.ones_like(current, dtype=torch.bool))
        observed = checked(item.trajectory.request.sample, tokens, advantage=item.advantage,
                           count=item.trajectory.tokens.shape[-1] - item.trajectory.prompt_length)
        evaluated, objective, reward = cotangents(observed, batch.profile, total=count, device=current.device,
                                                  linearized=linearized)
        observations.append(evaluated)
        objective_values, reward_values = parameter_vjps(linearized, trainable, objective=objective, reward=reward)
        reward_gradients = tuple(total + addition for total, addition in zip(reward_gradients, reward_values, strict=True))
        accumulate_objective(trainable, objective_values)
    mean = number(loss(observations))
    norm = gradient_norm(tuple(value.grad for value in trainable))
    reward_norm = gradient_norm(reward_gradients)
    recorded = capture_gradients(model, reward_gradients)
    optimizer.step()
    check_optimizer(optimizer)
    if any(not value.isfinite().all() for value in parameters(model)):
        raise RuntimeError("Non-finite learner parameter after update")
    model.eval()
    summary = {"loss": mean, "gradient_norm": norm, "reward_gradient_norm": reward_norm, "active_tokens": count,
               "before": before, "after": digest(adapter_state(model)),
               "nonzero_advantages": sum(item.advantage != 0 for item in samples)}
    return Result(summary=summary, gradients=recorded, probabilities=tuple(observations))
