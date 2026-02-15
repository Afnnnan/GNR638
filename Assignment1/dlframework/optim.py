from __future__ import annotations

from array import array
from typing import Iterable

from .layers import Parameter


class SGD:
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity = [array("f", [0.0]) * len(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for pi, p in enumerate(self.params):
            v = self.velocity[pi]
            for i in range(len(p.data)):
                g = p.grad[i]
                if self.weight_decay != 0.0:
                    g += self.weight_decay * p.data[i]
                if self.momentum != 0.0:
                    v[i] = self.momentum * v[i] + g
                    g = v[i]
                p.data[i] -= self.lr * g
