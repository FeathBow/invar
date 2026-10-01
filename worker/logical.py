from collections.abc import Callable
from dataclasses import dataclass, field
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
    reference_source: str = "engine"
    reference: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Update(Generic[Tensor]):
    summary: dict
    gradients: dict[str, Tensor]
    proximal: dict[str, tuple[int, ...]]
    currents: tuple[tuple[int, str, tuple[int, ...]], ...]
    reference: dict[str, tuple[int, ...]] = field(default_factory=dict)


def plan(request, trajectories, exchange, reference=None):
    return Plan(trajectories={item.request.sample: item for item in trajectories}, steps=request.steps,
                nonzero=sum(item.advantage_bits & 0x7FFFFFFF != 0 for item in request.samples), exchange=exchange,
                reference_source=request.reference_source, reference={} if reference is None else reference)
