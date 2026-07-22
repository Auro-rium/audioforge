from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    epoch: int = 0,
    global_step: int = 0,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save a portable single-file checkpoint with reproducibility metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config or {},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, output)
    return output


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid AudioForge checkpoint: {checkpoint}")
    return payload


def write_checkpoint_manifest(path: str | Path, checkpoint: str | Path, metadata: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(checkpoint), **metadata}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
