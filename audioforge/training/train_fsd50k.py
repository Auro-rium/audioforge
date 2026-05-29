from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from audioforge.training.trainer import FSD50KTrainer, load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FSD50K model with Accelerate.")

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML training config.",
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to Accelerate checkpoint directory.",
    )

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])

    return parser.parse_args()


def apply_cli_overrides(config, args: argparse.Namespace):
    updates = {
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "output_dir": args.output_dir,
        "mixed_precision": args.mixed_precision,
    }

    current = asdict(config)

    for key, value in updates.items():
        if value is not None:
            current[key] = value

    if args.output_dir is not None:
        current["checkpoint_dir"] = str(Path(args.output_dir) / "checkpoints")

    return type(config)(**current)


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    config = apply_cli_overrides(config, args)

    trainer = FSD50KTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()