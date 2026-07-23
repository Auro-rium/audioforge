#!/usr/bin/env python3
"""Push a trained AudioForge FSD50K checkpoint to the Hugging Face Hub.

Auth: run `huggingface-cli login` (or set the HF_TOKEN env var) on the
machine you run this from *before* invoking it. Never pass a token as a
CLI argument or paste it into a chat/terminal you don't control -- both
end up in shell history / logs.

scratch_cnn is a custom architecture with no native `transformers` class,
so it's published as config.json + model.safetensors + a README with a
loading snippet, uploaded via a plain HfApi().upload_folder() call.

ast is published through peft's built-in PeftModel.push_to_hub() (or, if
trained without LoRA, transformers' own PreTrainedModel.push_to_hub()) --
both already handle the adapter/model card export correctly, no custom
repo-building code needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

from audioforge.models.ast import create_ast_classifier
from audioforge.models.scratch_cnn import create_scratch_cnn

SCRATCH_CNN_README_TEMPLATE = """---
license: mit
library_name: audioforge
pipeline_tag: audio-classification
tags:
  - audio-classification
  - fsd50k
  - cnn
---

# {repo_id}

`ScratchAudioCNN` trained from random initialization on FSD50K for
multi-label environmental sound event classification, produced by the
[AudioForge](https://github.com/Auro-rium/audioforge) training pipeline.

## Validation metrics (FSD50K)

| Metric | Value |
|---|---|
| mAP | {mAP} |
| micro F1 | {micro_f1} |
| macro F1 | {macro_f1} |

## Usage

This is a custom PyTorch architecture, not a native `transformers` class --
loading it requires the `audioforge` package:

```python
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from audioforge.models.scratch_cnn import create_scratch_cnn

config_path = hf_hub_download("{repo_id}", "config.json")
weights_path = hf_hub_download("{repo_id}", "model.safetensors")

with open(config_path, encoding="utf-8") as f:
    config = json.load(f)

model = create_scratch_cnn(**config)
model.load_state_dict(load_file(weights_path))
model.eval()
```

Input: log-mel spectrogram `[batch, 1, n_mels, frames]` (see
`audioforge.features.logmel`). Output: raw logits `[batch, num_labels]`;
apply `torch.sigmoid` for per-label probabilities.
"""


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Not an AudioForge checkpoint (missing model_state_dict): {checkpoint_path}")

    return payload


def export_scratch_cnn(
    checkpoint_path: Path,
    repo_id: str,
    *,
    private: bool,
    dry_run: bool,
    export_dir: Path,
) -> None:
    payload = _load_checkpoint(checkpoint_path)
    extra = payload.get("extra", {}) if isinstance(payload.get("extra"), dict) else {}
    train_config = extra.get("config", {}) if isinstance(extra.get("config"), dict) else {}
    metrics = extra.get("metrics", {}) if isinstance(extra.get("metrics"), dict) else {}

    scratch_config = {
        "num_labels": int(train_config.get("num_labels", 200)),
        "in_channels": 1,
        "base_channels": int(train_config.get("base_channels", 32)),
        "dropout": float(train_config.get("dropout", 0.2)),
    }

    # Round-trip through the real constructor once, locally, so a shape
    # mismatch between the checkpoint and this config is caught here --
    # before anything is uploaded -- rather than surfacing later for
    # whoever downloads it.
    model = create_scratch_cnn(**scratch_config)
    model.load_state_dict(payload["model_state_dict"])

    export_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), export_dir / "model.safetensors")
    (export_dir / "config.json").write_text(json.dumps(scratch_config, indent=2), encoding="utf-8")

    readme = SCRATCH_CNN_README_TEMPLATE.format(
        repo_id=repo_id,
        mAP=metrics.get("mAP", "n/a"),
        micro_f1=metrics.get("micro_f1", "n/a"),
        macro_f1=metrics.get("macro_f1", "n/a"),
    )
    (export_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[export] wrote local export to {export_dir}")

    if dry_run:
        print("[export] --dry-run set, not pushing to the Hub")
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(export_dir), repo_id=repo_id)
    print(f"[export] pushed to https://huggingface.co/{repo_id}")


def export_ast(
    checkpoint_path: Path,
    repo_id: str,
    *,
    private: bool,
    dry_run: bool,
) -> None:
    payload = _load_checkpoint(checkpoint_path)
    extra = payload.get("extra", {}) if isinstance(payload.get("extra"), dict) else {}
    train_config = extra.get("config", {}) if isinstance(extra.get("config"), dict) else {}

    use_lora = bool(train_config.get("use_lora", False))

    model = create_ast_classifier(
        pretrained_name_or_path=train_config.get(
            "pretrained_name_or_path", "MIT/ast-finetuned-audioset-10-10-0.4593"
        ),
        num_labels=int(train_config.get("num_labels", 200)),
        dropout=float(train_config.get("dropout", 0.1)),
        freeze_backbone=bool(train_config.get("freeze_backbone", False)),
        use_lora=use_lora,
        lora_r=int(train_config.get("lora_r", 8)),
        lora_alpha=int(train_config.get("lora_alpha", 16)),
        lora_dropout=float(train_config.get("lora_dropout", 0.05)),
        lora_target_modules=tuple(train_config.get("lora_target_modules", ["query", "value"])),
    )
    # LoRA checkpoints only contain adapter + classifier weights (peft's
    # state_dict() is reduced by design); strict=False is expected there,
    # not a sign of a broken checkpoint. Full fine-tune checkpoints load
    # strictly.
    model.model.load_state_dict(payload["model_state_dict"], strict=not use_lora)

    if dry_run:
        kind = "LoRA adapter" if use_lora else "full model"
        print(f"[export] --dry-run set, would push {kind} -> {repo_id}")
        return

    # model.model is either a peft.PeftModel (use_lora=True) or a plain
    # transformers PreTrainedModel (full fine-tune / frozen backbone) --
    # both implement push_to_hub natively, so no custom repo-building
    # code is needed for either case.
    model.model.push_to_hub(repo_id, private=private)
    print(f"[export] pushed to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push a trained AudioForge FSD50K checkpoint to the Hugging Face Hub."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a *_best.pt checkpoint produced by FSD50KTrainer.",
    )
    parser.add_argument("--model-type", choices=["scratch_cnn", "ast"], required=True)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="e.g. auro-rirum/audioforge-scratch-cnn-fsd50k",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build/validate the export locally without pushing to the Hub.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("outputs/hf_export/scratch_cnn"),
        help="scratch_cnn only: local staging directory for config.json/model.safetensors/README.md.",
    )
    args = parser.parse_args()

    if args.model_type == "scratch_cnn":
        export_scratch_cnn(
            args.checkpoint,
            args.repo_id,
            private=args.private,
            dry_run=args.dry_run,
            export_dir=args.export_dir,
        )
    else:
        export_ast(
            args.checkpoint,
            args.repo_id,
            private=args.private,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
