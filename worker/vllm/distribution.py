from dataclasses import dataclass

import torch
from torch.overrides import TorchFunctionMode

from worker.distribution import Snapshot
from worker.probeschema import MASS_CAPTURE as CAPTURE


@dataclass(frozen=True, kw_only=True)
class Position:
    row: int
    step: int

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.row, self.step)):
            raise ValueError("Mass capture requires nonnegative row and response-step indices")


def batch_shape(shape):
    if type(shape) is not tuple or len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("Mass capture requires a positive row/vocabulary shape")


def selection(shape, positions):
    batch_shape(shape)
    if type(positions) is not tuple or not positions or any(not isinstance(value, Position) for value in positions):
        raise ValueError("Mass capture requires immutable selected positions")
    rows = tuple(value.row for value in positions)
    if len(set(rows)) != len(rows) or max(rows) >= shape[0]:
        raise ValueError("Mass capture selected rows are repeated or outside the native batch")


def snapshot(values, step):
    encoded = values.detach().cpu().contiguous().view(torch.int32).tolist()
    return Snapshot(step=step, probability_bits=tuple(int(word) & 0xffffffff for word in encoded))


def dispatched(func, args=(), kwargs=None, *, capture):
    supplied = {} if kwargs is None else kwargs
    result = func(*args, **supplied)
    if func is torch.Tensor.softmax:
        dimension = supplied.get("dim", args[1] if len(args) > 1 else None)
        capture.observe(result, dimension=dimension)
    return result


class _Masses(TorchFunctionMode):
    def __init__(self, shape, positions):
        super().__init__()
        selection(shape, positions)
        self.shape = shape
        self.positions = positions
        self.snapshots = None

    def __torch_function__(self, func, types, *args, **kwargs):
        return dispatched(func, *args, **kwargs, capture=self)

    def observe(self, result, *, dimension):
        if self.snapshots is not None:
            raise ValueError("Mass capture observed multiple softmax operations in one sampler call")
        if type(dimension) is not int or dimension not in (-1, 1):
            raise ValueError("Mass capture requires softmax over the vocabulary dimension")
        if result.dtype != torch.float32 or tuple(result.shape) != self.shape:
            raise ValueError("Mass capture output differs from the declared FP32 native shape")
        # Native sampling may overwrite this tensor; retain only selected words now.
        self.snapshots = tuple((position, snapshot(result[position.row], position.step)) for position in self.positions)

    def completed(self):
        if self.snapshots is None:
            raise ValueError("Mass capture did not observe an actual Tensor.softmax output")
        return self.snapshots


def capture(operation, *, shape, positions):
    observed = _Masses(shape, positions)
    with observed:
        result = operation()
    return result, observed.completed()
