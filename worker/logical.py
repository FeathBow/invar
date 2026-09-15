from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from worker.record import Evaluated
from worker.scalar import Profile
from worker.trajectory import Trajectory

Tensor = TypeVar("Tensor")
Model = TypeVar("Model")
Optimizer = TypeVar("Optimizer")


@dataclass(frozen=True, kw_only=True)
class Sample(Generic[Tensor]):
    trajectory: Trajectory[Tensor]
    proximal: Tensor
    reference: Tensor
    advantage: float


@dataclass(frozen=True, kw_only=True)
class Batch(Generic[Tensor]):
    samples: tuple[Sample, ...]
    order: tuple[str, ...]
    profile: Profile


@dataclass(frozen=True, kw_only=True)
class Learner(Generic[Model, Optimizer, Tensor]):
    model: Model
    optimizer: Optimizer
    evaluate: Callable[[Model, Trajectory], Tensor]


@dataclass(frozen=True, kw_only=True)
class Result(Generic[Tensor]):
    summary: dict
    gradients: dict[str, Tensor]
    probabilities: tuple[Evaluated, ...]


def ordered(batch):
    if not batch.order or any(not name for name in batch.order) or len(set(batch.order)) != len(batch.order):
        raise ValueError("Logical batch order must contain distinct sample identities")
    samples = {item.trajectory.request.sample: item for item in batch.samples}
    if len(samples) != len(batch.samples) or set(samples) != set(batch.order):
        raise ValueError("Delivered samples do not match the declared logical batch")
    return tuple(samples[name] for name in batch.order)
