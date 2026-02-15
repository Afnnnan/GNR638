from __future__ import annotations

import math
from array import array
from typing import List, Tuple

from .tensor import Tensor


def softmax_cross_entropy_with_logits(logits: Tensor, targets: List[int]) -> Tuple[float, Tensor]:
    n, c = logits.shape
    probs = array("f", [0.0]) * (n * c)
    loss = 0.0

    for i in range(n):
        row_offset = i * c
        max_logit = logits.data[row_offset]
        for j in range(1, c):
            v = logits.data[row_offset + j]
            if v > max_logit:
                max_logit = v

        denom = 0.0
        for j in range(c):
            e = math.exp(logits.data[row_offset + j] - max_logit)
            probs[row_offset + j] = e
            denom += e

        for j in range(c):
            probs[row_offset + j] /= denom

        p = max(1e-12, probs[row_offset + targets[i]])
        loss += -math.log(p)

    loss /= n

    grad = Tensor(probs, (n, c), requires_grad=False)
    inv_n = 1.0 / n
    for i in range(n):
        grad.data[i * c + targets[i]] -= 1.0
    for i in range(len(grad.data)):
        grad.data[i] *= inv_n

    return loss, grad
