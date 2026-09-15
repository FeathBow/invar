from dataclasses import dataclass
from typing import Generic, TypeVar

Tensor = TypeVar("Tensor")


@dataclass(frozen=True, kw_only=True)
class Request:
    sample: str
    group: str
    prompt: str
    seed: int
    limit: int
    temperature: float


@dataclass(frozen=True, kw_only=True)
class Trajectory(Generic[Tensor]):
    request: Request
    tokens: Tensor
    prompt_length: int
    behavior: Tensor
    text: str
    truncated: bool
