from __future__ import annotations

from typing import Any

import torch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved[key] = {
                inner_key: inner_value.to(device, non_blocking=True) if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def build_optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def format_metrics(metrics: dict[str, torch.Tensor]) -> str:
    pieces = []
    for key, value in metrics.items():
        scalar = value.item() if torch.is_tensor(value) else float(value)
        pieces.append(f"{key}={scalar:.4f}")
    return " ".join(pieces)
