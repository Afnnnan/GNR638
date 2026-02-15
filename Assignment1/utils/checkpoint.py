from __future__ import annotations

from typing import Dict

from dlframework.layers import Parameter, Sequential
from .config import save_json, load_json


def save_checkpoint(path: str, model: Sequential, metadata: Dict) -> None:
    params = model.parameters()
    state = {
        "metadata": metadata,
        "params": [list(p.data) for p in params],
        "shapes": [list(p.shape) for p in params],
    }
    save_json(path, state)


def load_checkpoint(path: str, model: Sequential) -> Dict:
    state = load_json(path)
    params = model.parameters()
    saved_params = state["params"]
    if len(params) != len(saved_params):
        raise ValueError("Checkpoint parameter count mismatch")

    for p, saved in zip(params, saved_params):
        if len(p.data) != len(saved):
            raise ValueError("Checkpoint tensor size mismatch")
        for i, v in enumerate(saved):
            p.data[i] = float(v)

    return state.get("metadata", {})
