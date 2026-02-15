from __future__ import annotations

import ctypes
import os
import sys
from array import array
from pathlib import Path
from typing import Tuple


class CppBackend:
    def __init__(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = self._default_library_path()
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"C++ backend library not found: {lib_path}. Build it first (python3 build_backend.py)."
            )
        self.lib = ctypes.CDLL(lib_path)
        self._bind()

    def _default_library_path(self) -> str:
        root = Path(__file__).resolve().parents[1]
        name = "libgnrbackend.dylib" if sys.platform == "darwin" else ("gnrbackend.dll" if sys.platform.startswith("win") else "libgnrbackend.so")
        return str(root / "build" / name)

    def _bind(self) -> None:
        self.lib.relu_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.relu_backward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        self.lib.linear_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.linear_backward_input.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.linear_backward_weight.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.linear_backward_bias.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.lib.maxpool2d_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.maxpool2d_backward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.lib.conv2d_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.conv2d_backward_input.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.conv2d_backward_weight.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.conv2d_backward_bias.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

    def _fptr(self, arr: array) -> ctypes.POINTER(ctypes.c_float):
        return (ctypes.c_float * len(arr)).from_buffer(arr)

    def _iptr(self, arr: array) -> ctypes.POINTER(ctypes.c_int32):
        return (ctypes.c_int32 * len(arr)).from_buffer(arr)

    def relu_forward(self, x: array, y: array) -> None:
        self.lib.relu_forward(self._fptr(x), self._fptr(y), len(x))

    def relu_backward(self, x: array, grad_out: array, grad_x: array) -> None:
        self.lib.relu_backward(self._fptr(x), self._fptr(grad_out), self._fptr(grad_x), len(x))

    def linear_forward(self, x: array, w: array, b: array, y: array, batch: int, in_features: int, out_features: int) -> None:
        self.lib.linear_forward(
            self._fptr(x), self._fptr(w), self._fptr(b), self._fptr(y), batch, in_features, out_features
        )

    def linear_backward_input(self, grad_out: array, w: array, grad_x: array, batch: int, in_features: int, out_features: int) -> None:
        self.lib.linear_backward_input(
            self._fptr(grad_out), self._fptr(w), self._fptr(grad_x), batch, in_features, out_features
        )

    def linear_backward_weight(self, x: array, grad_out: array, grad_w: array, batch: int, in_features: int, out_features: int) -> None:
        self.lib.linear_backward_weight(
            self._fptr(x), self._fptr(grad_out), self._fptr(grad_w), batch, in_features, out_features
        )

    def linear_backward_bias(self, grad_out: array, grad_b: array, batch: int, out_features: int) -> None:
        self.lib.linear_backward_bias(self._fptr(grad_out), self._fptr(grad_b), batch, out_features)

    def maxpool2d_forward(
        self,
        x: array,
        y: array,
        indices: array,
        batch: int,
        channels: int,
        in_h: int,
        in_w: int,
        kernel: int,
        stride: int,
    ) -> None:
        self.lib.maxpool2d_forward(
            self._fptr(x), self._fptr(y), self._iptr(indices), batch, channels, in_h, in_w, kernel, stride
        )

    def maxpool2d_backward(self, grad_out: array, indices: array, grad_x: array) -> None:
        self.lib.maxpool2d_backward(self._fptr(grad_out), self._iptr(indices), self._fptr(grad_x), len(grad_out), len(grad_x))

    def conv2d_forward(
        self,
        x: array,
        w: array,
        b: array,
        y: array,
        batch: int,
        in_channels: int,
        in_h: int,
        in_w: int,
        out_channels: int,
        kernel_h: int,
        kernel_w: int,
        stride: int,
        padding: int,
    ) -> None:
        self.lib.conv2d_forward(
            self._fptr(x),
            self._fptr(w),
            self._fptr(b),
            self._fptr(y),
            batch,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )

    def conv2d_backward_input(
        self,
        grad_out: array,
        w: array,
        grad_x: array,
        batch: int,
        in_channels: int,
        in_h: int,
        in_w: int,
        out_channels: int,
        kernel_h: int,
        kernel_w: int,
        stride: int,
        padding: int,
    ) -> None:
        self.lib.conv2d_backward_input(
            self._fptr(grad_out),
            self._fptr(w),
            self._fptr(grad_x),
            batch,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )

    def conv2d_backward_weight(
        self,
        x: array,
        grad_out: array,
        grad_w: array,
        batch: int,
        in_channels: int,
        in_h: int,
        in_w: int,
        out_channels: int,
        kernel_h: int,
        kernel_w: int,
        stride: int,
        padding: int,
    ) -> None:
        self.lib.conv2d_backward_weight(
            self._fptr(x),
            self._fptr(grad_out),
            self._fptr(grad_w),
            batch,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )

    def conv2d_backward_bias(self, grad_out: array, grad_b: array, batch: int, out_channels: int, out_h: int, out_w: int) -> None:
        self.lib.conv2d_backward_bias(self._fptr(grad_out), self._fptr(grad_b), batch, out_channels, out_h, out_w)
