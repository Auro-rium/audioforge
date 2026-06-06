from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FSD50K benchmark row.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_last_logged_loss(log_path: Path) -> float | None:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"loss=([0-9]+\.[0-9]+)", text)

    if not matches:
        return None

    return float(matches[-1])


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()

    metrics_path = args.run_dir / "best" / "best_metrics.json"
    config_path = args.run_dir / "train_config.json"
    distributed_path = args.run_dir / "distributed.json"
    log_path = args.run_dir / "train_stdout.log"

    if not metrics_path.exists():
        raise SystemExit(f"missing metrics: {metrics_path}")

    if not config_path.exists():
        raise SystemExit(f"missing config: {config_path}")

    metrics = load_json(metrics_path)
    config = load_json(config_path)
    distributed = load_json(distributed_path) if distributed_path.exists() else {}

    model_name = args.model_name or config.get("model_name", "unknown")
    checkpoint_name = f"{model_name}_best.pt"

    row = {
        "model": model_name,
        "pretrained_name_or_path": config.get("pretrained_name_or_path"),
        "dataset": "FSD50K",
        "classes": int(config.get("num_labels", 200)),
        "train_manifest": config.get("train_manifest"),
        "val_manifest": config.get("val_manifest"),
        "epochs": config.get("epochs"),
        "batch_size_per_process": config.get("batch_size"),
        "eval_batch_size_per_process": config.get("eval_batch_size"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "effective_batch_size": (
            int(config.get("batch_size", 0))
            * int(config.get("gradient_accumulation_steps", 1))
            * int(distributed.get("num_processes", 1) or 1)
        ),
        "num_processes": distributed.get("num_processes"),
        "mixed_precision": distributed.get("mixed_precision"),
        "learning_rate": config.get("learning_rate"),
        "mAP": metrics.get("mAP"),
        "micro_average_precision": metrics.get("micro_average_precision"),
        "macro_f1": metrics.get("macro_f1"),
        "micro_f1": metrics.get("micro_f1"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "micro_precision": metrics.get("micro_precision"),
        "micro_recall": metrics.get("micro_recall"),
        "threshold": metrics.get("threshold"),
        "num_val_samples": metrics.get("num_samples"),
        "num_labels": metrics.get("num_labels"),
        "last_logged_loss": parse_last_logged_loss(log_path),
        "checkpoint": str(args.run_dir / "best" / checkpoint_name),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    markdown = (
        "| Model | Pretrained | Dataset | Classes | mAP | Micro AP | Macro-F1 | Micro-F1 | "
        "Val Samples | Processes | Precision | Effective Batch | Checkpoint |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|\n"
        f"| {row['model']} | {row['pretrained_name_or_path']} | {row['dataset']} | "
        f"{row['classes']} | {fmt(row['mAP'])} | {fmt(row['micro_average_precision'])} | "
        f"{fmt(row['macro_f1'])} | {fmt(row['micro_f1'])} | "
        f"{row['num_val_samples']} | {row['num_processes']} | {row['mixed_precision']} | "
        f"{row['effective_batch_size']} | `{row['checkpoint']}` |\n"
    )

    args.markdown_out.write_text(markdown, encoding="utf-8")

    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
