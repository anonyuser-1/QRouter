from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    output_dir: str | Path,
    model,
    optimizer,
    scheduler,
    step: int,
    stage: str,
    args_dict: dict,
    scaler=None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint-step-{step}.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "stage": stage,
        "args": args_dict,
    }
    torch.save(payload, checkpoint_path)
    latest_path = output_dir / "latest.pt"
    torch.save(payload, latest_path)
    return checkpoint_path


def maybe_load_checkpoint(
    checkpoint_path: Optional[str | Path],
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> int:
    if checkpoint_path is None:
        return 0
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=False)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("step", 0))


def save_lora_adapter(model, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = getattr(model.llm_backbone, "llm", None)
    if llm is not None and hasattr(llm, "save_pretrained"):
        llm.save_pretrained(output_dir / "llm_lora")
    prompt_encoder = getattr(getattr(model.grounding_module, "prompt_encoder", None), "backbone", None)
    if prompt_encoder is not None and hasattr(prompt_encoder, "save_pretrained"):
        prompt_encoder.save_pretrained(output_dir / "grounding_qwen_lora")
