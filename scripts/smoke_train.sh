#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[smoke] AudioForge smoke training started"

if [ ! -f "pyproject.toml" ]; then
  echo "[smoke][error] pyproject.toml not found. Run from repo root or keep script in scripts/."
  exit 1
fi

if [ ! -f "configs/fsd50k/smoke.yaml" ]; then
  echo "[smoke][error] configs/fsd50k/smoke.yaml not found"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/train.csv" ]; then
  echo "[smoke][error] data/manifests/fsd50k/train.csv not found"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/val.csv" ]; then
  echo "[smoke][error] data/manifests/fsd50k/val.csv not found"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/label_map.json" ]; then
  echo "[smoke][error] data/manifests/fsd50k/label_map.json not found"
  exit 1
fi

python - <<'PY'
import audioforge
from audioforge.training.trainer import FSD50KTrainConfig, load_train_config
from audioforge.models.scratch_cnn import create_scratch_cnn
from audioforge.data.manifests import read_manifest

cfg = load_train_config("configs/fsd50k/smoke.yaml")
train = read_manifest(cfg.train_manifest)
val = read_manifest(cfg.val_manifest)
model = create_scratch_cnn(num_labels=cfg.num_labels, base_channels=cfg.base_channels)

print("[smoke] import ok")
print("[smoke] train rows:", len(train))
print("[smoke] val rows:", len(val))
print("[smoke] model:", model.__class__.__name__)
print("[smoke] config output:", cfg.output_dir)
PY

rm -rf outputs/smoke_scratch

python -m audioforge.training.train_fsd50k \
  --config configs/fsd50k/smoke.yaml

echo "[smoke] checking outputs"

test -f outputs/smoke_scratch/train_config.json
test -f outputs/smoke_scratch/distributed.json
test -f outputs/smoke_scratch/best/scratch_cnn_best.pt
test -f outputs/smoke_scratch/best/best_metrics.json

python - <<'PY'
import json
from pathlib import Path

metrics_path = Path("outputs/smoke_scratch/best/best_metrics.json")
with metrics_path.open("r", encoding="utf-8") as f:
    metrics = json.load(f)

required = [
    "mAP",
    "micro_average_precision",
    "macro_f1",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "micro_precision",
    "micro_recall",
    "num_samples",
    "num_labels",
]

missing = [key for key in required if key not in metrics]
if missing:
    raise SystemExit(f"[smoke][error] missing metric keys: {missing}")

print("[smoke] best mAP:", metrics["mAP"])
print("[smoke] micro F1:", metrics["micro_f1"])
print("[smoke] num samples:", metrics["num_samples"])
print("[smoke] num labels:", metrics["num_labels"])
print("[smoke] metrics ok")
PY

echo "[smoke] generated files:"
find outputs/smoke_scratch -maxdepth 3 -type f | sort

echo "[smoke] SMOKE TRAINING PASSED ✅"
