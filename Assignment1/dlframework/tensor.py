from __future__ import annotations

from array import array
from typing import Iterable, List, Sequence, Tuple


class Tensor:
    def __init__(self, data: array, shape: Sequence[int], requires_grad: bool = False):
        self.data = data
        self.shape = tuple(shape)
        self.requires_grad = requires_grad
        self.grad = array("f", [0.0]) * len(self.data) if requires_grad else None

    @property
    def size(self) -> int:
        return len(self.data)

    def zero_grad(self) -> None:
        if self.grad is not None:
            for i in range(len(self.grad)):
                self.grad[i] = 0.0


def prod(shape: Sequence[int]) -> int:
    p = 1
    for d in shape:
        p *= d
    return p


def zeros(shape: Sequence[int], requires_grad: bool = False) -> Tensor:
    return Tensor(array("f", [0.0]) * prod(shape), shape, requires_grad=requires_grad)


def from_list(values: Iterable[float], shape: Sequence[int], requires_grad: bool = False) -> Tensor:
    arr = array("f", values)
    if len(arr) != prod(shape):
        raise ValueError(f"shape {shape} does not match data length {len(arr)}")
    return Tensor(arr, shape, requires_grad=requires_grad)


def flatten_batch(t: Tensor) -> Tensor:
    if len(t.shape) < 2:
        raise ValueError("flatten_batch requires batch dimension")
    batch = t.shape[0]
    features = 1
    for d in t.shape[1:]:
        features *= d
    out = Tensor(array("f", t.data), (batch, features), requires_grad=t.requires_grad)
    out._parent = t
    return out


def unflatten_batch_like(flat: Tensor, like: Tensor) -> Tensor:
    out = Tensor(array("f", flat.data), like.shape, requires_grad=flat.requires_grad)
    return out
