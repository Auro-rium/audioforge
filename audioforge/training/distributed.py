from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


def create_accelerator(
    *,
    output_dir: str | Path,
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 1,
) -> Accelerator:
    """Create Hugging Face Accelerator for single/multi-GPU training."""

    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(Path(output_dir) / "logs"),
        automatic_checkpoint_naming=False,
    )

    return Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
    )


def setup_torch_runtime(*, deterministic: bool = False) -> None:
    """Configure PyTorch runtime behavior."""

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    torch.set_float32_matmul_precision("high")


def is_main_process(accelerator: Accelerator) -> bool:
    return bool(accelerator.is_main_process)


def get_distributed_info(accelerator: Accelerator) -> dict[str, Any]:
    return {
        "device": str(accelerator.device),
        "num_processes": accelerator.num_processes,
        "process_index": accelerator.process_index,
        "local_process_index": accelerator.local_process_index,
        "is_main_process": accelerator.is_main_process,
        "mixed_precision": accelerator.mixed_precision,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_model_state(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save unwrapped model state dict on main process only."""

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    payload = {
        "model_state_dict": state_dict,
        "model_class": unwrapped.__class__.__name__,
        "extra": extra or {},
    }

    accelerator.save(payload, str(output_path))