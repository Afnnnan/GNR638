#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def to_points(values, width=800, height=300, pad=30):
    if not values:
        return ""
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    pts = []
    n = len(values)
    for i, v in enumerate(values):
        x = pad + (width - 2 * pad) * (i / max(1, n - 1))
        y = height - pad - (height - 2 * pad) * ((v - vmin) / (vmax - vmin))
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def write_svg(path, title, a_name, a_vals, b_name, b_vals):
    width = 900
    height = 360
    os.makedirs(os.path.dirname(path), exist_ok=True)

    a_points = to_points(a_vals, width=width, height=height)
    b_points = to_points(b_vals, width=width, height=height)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>
<line x1="50" y1="20" x2="50" y2="320" stroke="#222"/>
<line x1="50" y1="320" x2="860" y2="320" stroke="#222"/>
<text x="50" y="15" font-size="14" font-family="monospace">{title}</text>
<polyline fill="none" stroke="#006d77" stroke-width="2" points="{a_points}"/>
<polyline fill="none" stroke="#e76f51" stroke-width="2" points="{b_points}"/>
<text x="670" y="30" font-size="12" font-family="monospace" fill="#006d77">{a_name}</text>
<text x="670" y="48" font-size="12" font-family="monospace" fill="#e76f51">{b_name}</text>
</svg>'''

    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    p = argparse.ArgumentParser(description="Generate SVG training plots without third-party plotting libraries")
    p.add_argument("--log_path", required=True)
    p.add_argument("--out_dir", default="reports/plots")
    args = p.parse_args()

    with open(args.log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hist = data["history"]
    train_loss = [r["train_loss"] for r in hist]
    val_loss = [r["val_loss"] for r in hist]
    train_acc = [r["train_acc"] for r in hist]
    val_acc = [r["val_acc"] for r in hist]

    write_svg(os.path.join(args.out_dir, "loss.svg"), "Loss vs Epoch", "train_loss", train_loss, "val_loss", val_loss)
    write_svg(os.path.join(args.out_dir, "accuracy.svg"), "Accuracy vs Epoch", "train_acc", train_acc, "val_acc", val_acc)
    print("Saved:", os.path.join(args.out_dir, "loss.svg"))
    print("Saved:", os.path.join(args.out_dir, "accuracy.svg"))


if __name__ == "__main__":
    main()
