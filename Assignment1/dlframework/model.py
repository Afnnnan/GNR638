from __future__ import annotations

from array import array

from .backend import CppBackend
from .layers import Conv2D, Flatten, GlobalAvgPool2D, Linear, MaxPool2D, Module, ReLU, Sequential
from .tensor import Tensor


def build_cnn(
    backend: CppBackend,
    num_classes: int,
    in_channels: int = 3,
    in_size: int = 32,
    conv1_out: int = 8,
    conv2_out: int = 16,
    hidden: int = 64,
):
    h = in_size
    w = in_size

    h = (h + 2 * 1 - 3) // 1 + 1
    w = (w + 2 * 1 - 3) // 1 + 1
    h = h // 2
    w = w // 2

    h = (h + 2 * 1 - 3) // 1 + 1
    w = (w + 2 * 1 - 3) // 1 + 1
    h = h // 2
    w = w // 2

    flatten = conv2_out * h * w

    return Sequential(
        [
            Conv2D(backend, in_channels=in_channels, out_channels=conv1_out, kernel_size=3, stride=1, padding=1),
            ReLU(backend),
            MaxPool2D(backend, kernel_size=2, stride=2),
            Conv2D(backend, in_channels=conv1_out, out_channels=conv2_out, kernel_size=3, stride=1, padding=1),
            ReLU(backend),
            MaxPool2D(backend, kernel_size=2, stride=2),
            Flatten(),
            Linear(backend, in_features=flatten, out_features=hidden),
            ReLU(backend),
            Linear(backend, in_features=hidden, out_features=num_classes),
        ]
    )


class ResidualBlock(Module):
    def __init__(self, backend: CppBackend, in_channels: int, out_channels: int, stride: int = 1):
        self.conv1 = Conv2D(backend, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu1 = ReLU(backend)
        self.conv2 = Conv2D(backend, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU(backend)
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = Conv2D(backend, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        main = self.conv1.forward(x)
        main = self.relu1.forward(main)
        main = self.conv2.forward(main)

        skip = self.proj.forward(x) if self.proj is not None else x
        out = Tensor(array("f", [0.0]) * main.size, main.shape, requires_grad=main.requires_grad)
        for i in range(main.size):
            out.data[i] = main.data[i] + skip.data[i]

        out = self.relu2.forward(out)
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        grad = self.relu2.backward(grad_out)

        grad_main = Tensor(array("f", grad.data), grad.shape, requires_grad=False)
        grad_skip = Tensor(array("f", grad.data), grad.shape, requires_grad=False)

        grad_main = self.conv2.backward(grad_main)
        grad_main = self.relu1.backward(grad_main)
        grad_main = self.conv1.backward(grad_main)

        if self.proj is not None:
            grad_skip = self.proj.backward(grad_skip)

        grad_x = Tensor(array("f", [0.0]) * grad_main.size, grad_main.shape, requires_grad=False)
        for i in range(grad_main.size):
            grad_x.data[i] = grad_main.data[i] + grad_skip.data[i]
        return grad_x

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        if self.proj is not None:
            params.extend(self.proj.parameters())
        return params


class ResNetSmall(Module):
    def __init__(self, backend: CppBackend, num_classes: int, in_channels: int = 3, base_channels: int = 16):
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = Conv2D(backend, in_channels=in_channels, out_channels=c1, kernel_size=3, stride=1, padding=1)
        self.stem_relu = ReLU(backend)

        # Tiny ResNet for assignment runtime limits.
        self.blocks = [
            ResidualBlock(backend, c1, c1, stride=1),
            ResidualBlock(backend, c1, c2, stride=2),
            ResidualBlock(backend, c2, c3, stride=2),
        ]

        self.gap = GlobalAvgPool2D()
        self.fc = Linear(backend, in_features=c3, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.stem.forward(x)
        out = self.stem_relu.forward(out)
        for b in self.blocks:
            out = b.forward(out)
        out = self.gap.forward(out)
        out = self.fc.forward(out)
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        grad = self.fc.backward(grad_out)
        grad = self.gap.backward(grad)
        for b in reversed(self.blocks):
            grad = b.backward(grad)
        grad = self.stem_relu.backward(grad)
        grad = self.stem.backward(grad)
        return grad

    def parameters(self):
        params = []
        params.extend(self.stem.parameters())
        for b in self.blocks:
            params.extend(b.parameters())
        params.extend(self.fc.parameters())
        return params


def build_resnet_small(
    backend: CppBackend,
    num_classes: int,
    in_channels: int = 3,
    base_channels: int = 16,
):
    return ResNetSmall(backend, num_classes=num_classes, in_channels=in_channels, base_channels=base_channels)


def build_model(
    backend: CppBackend,
    num_classes: int,
    model_type: str = "cnn",
    in_channels: int = 3,
    in_size: int = 32,
    conv1_out: int = 8,
    conv2_out: int = 16,
    hidden: int = 64,
    resnet_base_channels: int = 16,
):
    m = model_type.lower()
    if m == "cnn":
        return build_cnn(
            backend,
            num_classes=num_classes,
            in_channels=in_channels,
            in_size=in_size,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            hidden=hidden,
        )
    if m == "resnet":
        return build_resnet_small(
            backend,
            num_classes=num_classes,
            in_channels=in_channels,
            base_channels=resnet_base_channels,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
