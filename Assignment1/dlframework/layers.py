from __future__ import annotations

import math
import random
from array import array
from typing import List, Tuple

from .backend import CppBackend
from .tensor import Tensor, flatten_batch, unflatten_batch_like


class Parameter(Tensor):
    def __init__(self, data: array, shape: Tuple[int, ...]):
        super().__init__(data, shape, requires_grad=True)


class Module:
    def parameters(self) -> List[Parameter]:
        return []

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


class Conv2D(Module):
    def __init__(self, backend: CppBackend, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.backend = backend
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        scale = math.sqrt(2.0 / fan_in)
        w = array("f", [(random.random() * 2.0 - 1.0) * scale for _ in range(out_channels * in_channels * kernel_size * kernel_size)])
        b = array("f", [0.0 for _ in range(out_channels)])
        self.weight = Parameter(w, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Parameter(b, (out_channels,))
        self._x = None
        self._out_shape = None

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        k = self.kernel_size
        oh = (h + 2 * self.padding - k) // self.stride + 1
        ow = (w + 2 * self.padding - k) // self.stride + 1
        out = Tensor(array("f", [0.0]) * (n * self.out_channels * oh * ow), (n, self.out_channels, oh, ow), requires_grad=x.requires_grad)
        self.backend.conv2d_forward(
            x.data,
            self.weight.data,
            self.bias.data,
            out.data,
            n,
            c,
            h,
            w,
            self.out_channels,
            k,
            k,
            self.stride,
            self.padding,
        )
        self._x = x
        self._out_shape = out.shape
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        x = self._x
        n, c, h, w = x.shape
        _, oc, oh, ow = grad_out.shape
        k = self.kernel_size

        grad_x = Tensor(array("f", [0.0]) * x.size, x.shape, requires_grad=False)
        self.backend.conv2d_backward_input(
            grad_out.data,
            self.weight.data,
            grad_x.data,
            n,
            c,
            h,
            w,
            oc,
            k,
            k,
            self.stride,
            self.padding,
        )

        grad_w = array("f", [0.0]) * len(self.weight.data)
        self.backend.conv2d_backward_weight(
            x.data,
            grad_out.data,
            grad_w,
            n,
            c,
            h,
            w,
            oc,
            k,
            k,
            self.stride,
            self.padding,
        )
        for i in range(len(self.weight.grad)):
            self.weight.grad[i] += grad_w[i]

        grad_b = array("f", [0.0]) * len(self.bias.data)
        self.backend.conv2d_backward_bias(grad_out.data, grad_b, n, oc, oh, ow)
        for i in range(len(self.bias.grad)):
            self.bias.grad[i] += grad_b[i]

        return grad_x

    def parameters(self) -> List[Parameter]:
        return [self.weight, self.bias]


class ReLU(Module):
    def __init__(self, backend: CppBackend):
        self.backend = backend
        self._x = None

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(array("f", [0.0]) * x.size, x.shape, requires_grad=x.requires_grad)
        self.backend.relu_forward(x.data, out.data)
        self._x = x
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        grad_x = Tensor(array("f", [0.0]) * grad_out.size, grad_out.shape, requires_grad=False)
        self.backend.relu_backward(self._x.data, grad_out.data, grad_x.data)
        return grad_x


class MaxPool2D(Module):
    def __init__(self, backend: CppBackend, kernel_size: int = 2, stride: int = 2):
        self.backend = backend
        self.kernel_size = kernel_size
        self.stride = stride
        self._x_shape = None
        self._indices = None

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        oh = (h - self.kernel_size) // self.stride + 1
        ow = (w - self.kernel_size) // self.stride + 1
        out = Tensor(array("f", [0.0]) * (n * c * oh * ow), (n, c, oh, ow), requires_grad=x.requires_grad)
        self._indices = array("i", [0]) * (n * c * oh * ow)
        self.backend.maxpool2d_forward(
            x.data,
            out.data,
            self._indices,
            n,
            c,
            h,
            w,
            self.kernel_size,
            self.stride,
        )
        self._x_shape = x.shape
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        grad_x = Tensor(array("f", [0.0]) * (self._x_shape[0] * self._x_shape[1] * self._x_shape[2] * self._x_shape[3]), self._x_shape, requires_grad=False)
        self.backend.maxpool2d_backward(grad_out.data, self._indices, grad_x.data)
        return grad_x


class Linear(Module):
    def __init__(self, backend: CppBackend, in_features: int, out_features: int):
        self.backend = backend
        self.in_features = in_features
        self.out_features = out_features

        scale = math.sqrt(2.0 / max(1, in_features))
        w = array("f", [(random.random() * 2.0 - 1.0) * scale for _ in range(out_features * in_features)])
        b = array("f", [0.0 for _ in range(out_features)])
        self.weight = Parameter(w, (out_features, in_features))
        self.bias = Parameter(b, (out_features,))
        self._x = None

    def forward(self, x: Tensor) -> Tensor:
        n, in_features = x.shape
        out = Tensor(array("f", [0.0]) * (n * self.out_features), (n, self.out_features), requires_grad=x.requires_grad)
        self.backend.linear_forward(x.data, self.weight.data, self.bias.data, out.data, n, in_features, self.out_features)
        self._x = x
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        x = self._x
        n, in_features = x.shape
        grad_x = Tensor(array("f", [0.0]) * x.size, x.shape, requires_grad=False)
        self.backend.linear_backward_input(grad_out.data, self.weight.data, grad_x.data, n, in_features, self.out_features)

        grad_w = array("f", [0.0]) * len(self.weight.data)
        self.backend.linear_backward_weight(x.data, grad_out.data, grad_w, n, in_features, self.out_features)
        for i in range(len(self.weight.grad)):
            self.weight.grad[i] += grad_w[i]

        grad_b = array("f", [0.0]) * len(self.bias.data)
        self.backend.linear_backward_bias(grad_out.data, grad_b, n, self.out_features)
        for i in range(len(self.bias.grad)):
            self.bias.grad[i] += grad_b[i]

        return grad_x

    def parameters(self) -> List[Parameter]:
        return [self.weight, self.bias]


class Flatten(Module):
    def __init__(self):
        self._in = None

    def forward(self, x: Tensor) -> Tensor:
        self._in = x
        return flatten_batch(x)

    def backward(self, grad_out: Tensor) -> Tensor:
        return unflatten_batch_like(grad_out, self._in)


class GlobalAvgPool2D(Module):
    def __init__(self):
        self._in_shape = None

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        out = Tensor(array("f", [0.0]) * (n * c), (n, c), requires_grad=x.requires_grad)
        area = float(h * w)
        for ni in range(n):
            for ci in range(c):
                base = ((ni * c + ci) * h) * w
                s = 0.0
                for k in range(h * w):
                    s += x.data[base + k]
                out.data[ni * c + ci] = s / area
        self._in_shape = x.shape
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        n, c, h, w = self._in_shape
        grad_x = Tensor(array("f", [0.0]) * (n * c * h * w), self._in_shape, requires_grad=False)
        area = float(h * w)
        for ni in range(n):
            for ci in range(c):
                g = grad_out.data[ni * c + ci] / area
                base = ((ni * c + ci) * h) * w
                for k in range(h * w):
                    grad_x.data[base + k] = g
        return grad_x


class Sequential(Module):
    def __init__(self, layers: List[Module]):
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> List[Parameter]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self) -> None:
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        for layer in self.layers:
            layer.eval()
