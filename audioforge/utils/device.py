from __future__ import annotations

from typing import Any

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a configured device and fail clearly for unavailable accelerators."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return resolved


def device_info(device: str = "auto") -> dict[str, Any]:
    resolved = resolve_device(device)
    payload: dict[str, Any] = {
        "device": str(resolved),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if resolved.type == "cuda":
        payload["cuda_device_name"] = torch.cuda.get_device_name(resolved)
        payload["cuda_capability"] = list(torch.cuda.get_device_capability(resolved))
    return payload
