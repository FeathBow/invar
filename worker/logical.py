from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from worker.exchange import Exchange
from worker.trajectory import Trajectory

Tensor = TypeVar("Tensor")
Model = TypeVar("Model")
Optimizer = TypeVar("Optimizer")


@dataclass(frozen=True, kw_only=True)
class Learner(Generic[Model, Optimizer, Tensor]):
    model: Model
    optimizer: Optimizer
    evaluate: Callable[[Model, Trajectory], Tensor]


@dataclass(frozen=True, kw_only=True)
class Plan(Generic[Tensor]):
    trajectories: dict[str, Trajectory[Tensor]]
    steps: tuple[tuple[str, ...], ...]
    nonzero: int
    exchange: Exchange


@dataclass(frozen=True, kw_only=True)
class Update(Generic[Tensor]):
    summary: dict
    gradients: dict[str, Tensor]
    proximal: dict[str, tuple[int, ...]]
    currents: tuple[tuple[int, str, tuple[int, ...]], ...]


def plan(request, trajectories, exchange):
    return Plan(trajectories={item.request.sample: item for item in trajectories}, steps=request.steps,
                nonzero=sum(item.advantage_bits & 0x7FFFFFFF != 0 for item in request.samples), exchange=exchange)
