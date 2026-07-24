"""Generate a markdown report with real training/eval graphs from a completed run.

Reads the artifacts a training run already produces (train_stdout.log,
metrics/eval_step_*.json, best/best_metrics.json) and turns them into PNG
plots plus a markdown report. No fabricated numbers -- everything here is
parsed directly from what the trainer wrote to disk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_LINE_RE = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+) batch=\d+ loss=(?P<loss>[\d.]+) lr=(?P<lr>[\d.]+)"
)
EVAL_LINE_RE = re.compile(
    r"eval epoch=(?P<epoch>\d+) step=(?P<step>\d+) loss=(?P<loss>[\d.]+) "
    r"mAP=(?P<mAP>[\d.]+) micro_f1=(?P<micro_f1>[\d.]+) macro_f1=(?P<macro_f1>[\d.]+)"
)


def parse_train_log(log_path: Path) -> list[dict[str, float]]:
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = STEP_LINE_RE.search(line)
        if match:
            rows.append(
                {
                    "step": int(match["step"]),
                    "loss": float(match["loss"]),
                    "lr": float(match["lr"]),
                }
            )
    return rows


def parse_eval_log(log_path: Path) -> list[dict[str, float]]:
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = EVAL_LINE_RE.search(line)
        if match:
            rows.append(
                {
                    "step": int(match["step"]),
                    "loss": float(match["loss"]),
                    "mAP": float(match["mAP"]),
                    "micro_f1": float(match["micro_f1"]),
                    "macro_f1": float(match["macro_f1"]),
                }
            )
    return rows


def plot_loss_curve(train_rows: list[dict], out_path: Path, title: str) -> None:
    steps = [r["step"] for r in train_rows]
    losses = [r["loss"] for r in train_rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, losses, color="#2563eb", linewidth=1.5, label="train loss")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    lrs = [r["lr"] for r in train_rows]
    ax2.plot(steps, lrs, color="#f59e0b", linewidth=1.0, alpha=0.7, label="learning rate")
    ax2.set_ylabel("learning rate", color="#f59e0b")
    ax2.tick_params(axis="y", labelcolor="#f59e0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eval_curve(eval_rows: list[dict], out_path: Path, title: str) -> None:
    steps = [r["step"] for r in eval_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, [r["mAP"] for r in eval_rows], marker="o", label="mAP", color="#16a34a")
    ax.plot(steps, [r["micro_f1"] for r in eval_rows], marker="o", label="micro F1", color="#2563eb")
    ax.plot(steps, [r["macro_f1"] for r in eval_rows], marker="o", label="macro F1", color="#dc2626")
    ax.set_xlabel("training step")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_ap(final_eval_json: Path, out_path: Path, title: str, top_n: int = 20) -> None:
    data = json.loads(final_eval_json.read_text(encoding="utf-8"))
    per_class = [row for row in data["per_class"] if row["average_precision"] is not None]
    per_class.sort(key=lambda row: row["average_precision"], reverse=True)

    top = per_class[:top_n]
    bottom = per_class[-top_n:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh([row["label_name"] for row in reversed(top)], [row["average_precision"] for row in reversed(top)], color="#16a34a")
    axes[0].set_title(f"Top {top_n} classes by AP")
    axes[0].set_xlabel("average precision")

    axes[1].barh([row["label_name"] for row in reversed(bottom)], [row["average_precision"] for row in reversed(bottom)], color="#dc2626")
    axes[1].set_title(f"Bottom {top_n} classes by AP")
    axes[1].set_xlabel("average precision")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown_report(
    *,
    model_name: str,
    best_metrics: dict,
    train_config: dict,
    eval_rows: list[dict],
    loss_png: str,
    eval_png: str,
    per_class_png: str,
    out_path: Path,
) -> None:
    lines = [
        f"# {model_name} — FSD50K training report",
        "",
        "Generated directly from the completed training run's saved artifacts "
        "(train_stdout.log, metrics/eval_step_*.json, best_metrics.json). "
        "Every number below is a real measurement, not an estimate.",
        "",
        "## Run configuration",
        "",
        f"- Model: `{train_config.get('model_name', model_name)}`",
        f"- Epochs: {train_config.get('epochs')}",
        f"- Batch size: {train_config.get('batch_size')}",
        f"- Learning rate: {train_config.get('learning_rate')}",
        f"- Mixed precision: {train_config.get('mixed_precision')}",
        f"- Train rows: {train_config.get('num_train_rows', 'n/a')}",
        f"- Val rows: {best_metrics.get('num_val_samples', best_metrics.get('num_samples', 'n/a'))}",
        "",
        "## Final metrics (best checkpoint)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mAP | {best_metrics['mAP']:.4f} |",
        f"| Micro Average Precision | {best_metrics['micro_average_precision']:.4f} |",
        f"| Macro F1 | {best_metrics['macro_f1']:.4f} |",
        f"| Micro F1 | {best_metrics['micro_f1']:.4f} |",
        f"| Macro Precision | {best_metrics['macro_precision']:.4f} |",
        f"| Macro Recall | {best_metrics['macro_recall']:.4f} |",
        f"| Micro Precision | {best_metrics['micro_precision']:.4f} |",
        f"| Micro Recall | {best_metrics['micro_recall']:.4f} |",
        "",
        "## Training curves",
        "",
        f"![training loss]({loss_png})",
        "",
        f"![eval metrics]({eval_png})",
        "",
        "## Per-class average precision (best/worst performing classes)",
        "",
        f"![per-class AP]({per_class_png})",
        "",
        "## Eval progression (raw numbers)",
        "",
        "| Step | Epoch | Eval Loss | mAP | Micro F1 | Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for row in eval_rows:
        lines.append(
            f"| {row['step']} | - | {row['loss']:.4f} | {row['mAP']:.4f} | "
            f"{row['micro_f1']:.4f} | {row['macro_f1']:.4f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_log = run_dir / "train_stdout.log"
    train_rows = parse_train_log(train_log)
    eval_rows = parse_eval_log(train_log)

    best_metrics = json.loads((run_dir / "best" / "best_metrics.json").read_text(encoding="utf-8"))
    train_config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))

    metrics_dir = run_dir / "metrics"
    final_eval_json = sorted(
        metrics_dir.glob("eval_step_*.json"), key=lambda p: int(p.stem.split("_")[-1])
    )[-1]

    loss_png = out_dir / f"{args.model_name}_loss_curve.png"
    eval_png = out_dir / f"{args.model_name}_eval_curve.png"
    per_class_png = out_dir / f"{args.model_name}_per_class_ap.png"

    plot_loss_curve(train_rows, loss_png, f"{args.model_name} — training loss / learning rate")
    plot_eval_curve(eval_rows, eval_png, f"{args.model_name} — eval mAP / F1 over training")
    plot_per_class_ap(final_eval_json, per_class_png, f"{args.model_name} — per-class average precision")

    report_path = out_dir / f"{args.model_name}_report.md"
    write_markdown_report(
        model_name=args.model_name,
        best_metrics=best_metrics,
        train_config=train_config,
        eval_rows=eval_rows,
        loss_png=loss_png.name,
        eval_png=eval_png.name,
        per_class_png=per_class_png.name,
        out_path=report_path,
    )

    print(f"[report] wrote {report_path}")
    print(f"[report] wrote {loss_png}")
    print(f"[report] wrote {eval_png}")
    print(f"[report] wrote {per_class_png}")


if __name__ == "__main__":
    main()
