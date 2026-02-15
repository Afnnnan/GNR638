from __future__ import annotations

from typing import Dict


# MACs and FLOPs are computed per single forward sample.
def cnn_complexity(
    in_channels: int,
    in_h: int,
    in_w: int,
    num_classes: int,
    conv1_out: int,
    conv2_out: int,
    hidden: int,
    k1: int = 3,
    k2: int = 3,
    p1: int = 1,
    p2: int = 1,
    s1: int = 1,
    s2: int = 1,
) -> Dict[str, int]:
    h1 = (in_h + 2 * p1 - k1) // s1 + 1
    w1 = (in_w + 2 * p1 - k1) // s1 + 1
    h1p = h1 // 2
    w1p = w1 // 2

    h2 = (h1p + 2 * p2 - k2) // s2 + 1
    w2 = (w1p + 2 * p2 - k2) // s2 + 1
    h2p = h2 // 2
    w2p = w2 // 2

    fc_in = conv2_out * h2p * w2p

    params = (
        conv1_out * in_channels * k1 * k1 + conv1_out
        + conv2_out * conv1_out * k2 * k2 + conv2_out
        + hidden * fc_in + hidden
        + num_classes * hidden + num_classes
    )

    conv1_macs = h1 * w1 * conv1_out * (in_channels * k1 * k1)
    conv2_macs = h2 * w2 * conv2_out * (conv1_out * k2 * k2)
    fc1_macs = hidden * fc_in
    fc2_macs = num_classes * hidden
    macs = conv1_macs + conv2_macs + fc1_macs + fc2_macs

    return {
        "params": int(params),
        "macs": int(macs),
        "flops": int(macs * 2),
        "flatten_features": int(fc_in),
    }


def _conv2d_out(hw: int, kernel: int, stride: int, padding: int) -> int:
    return (hw + 2 * padding - kernel) // stride + 1


def _conv2d_params(in_ch: int, out_ch: int, kernel: int) -> int:
    return out_ch * in_ch * kernel * kernel + out_ch


def _conv2d_macs(h: int, w: int, in_ch: int, out_ch: int, kernel: int) -> int:
    return h * w * out_ch * (in_ch * kernel * kernel)


def resnet_small_complexity(
    in_channels: int,
    in_h: int,
    in_w: int,
    num_classes: int,
    base_channels: int = 16,
) -> Dict[str, int]:
    c1 = base_channels
    c2 = base_channels * 2
    c3 = base_channels * 4

    params = 0
    macs = 0

    # Stem conv 3x3 stride1.
    h = _conv2d_out(in_h, 3, 1, 1)
    w = _conv2d_out(in_w, 3, 1, 1)
    params += _conv2d_params(in_channels, c1, 3)
    macs += _conv2d_macs(h, w, in_channels, c1, 3)

    # Tiny residual stages with downsample in later blocks.
    in_ch = c1
    stage_defs = [
        (c1, 1),
        (c2, 2),
        (c3, 2),
    ]

    for out_ch, stride in stage_defs:
        oh = _conv2d_out(h, 3, stride, 1)
        ow = _conv2d_out(w, 3, stride, 1)

        # conv1 in block
        params += _conv2d_params(in_ch, out_ch, 3)
        macs += _conv2d_macs(oh, ow, in_ch, out_ch, 3)

        # conv2 in block
        params += _conv2d_params(out_ch, out_ch, 3)
        macs += _conv2d_macs(oh, ow, out_ch, out_ch, 3)

        # projection 1x1 if shape/channel mismatch
        if stride != 1 or in_ch != out_ch:
            params += _conv2d_params(in_ch, out_ch, 1)
            macs += _conv2d_macs(oh, ow, in_ch, out_ch, 1)

        h, w, in_ch = oh, ow, out_ch

    # Classifier after global average pooling.
    params += num_classes * c3 + num_classes
    macs += num_classes * c3

    return {
        "params": int(params),
        "macs": int(macs),
        "flops": int(macs * 2),
        "flatten_features": int(c3),
    }
