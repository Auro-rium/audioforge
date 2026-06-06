#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_SAMPLES="${AUDIOFORGE_RANDOM_TRAIN_SAMPLES:-5000}"
VAL_SAMPLES="${AUDIOFORGE_RANDOM_VAL_SAMPLES:-500}"
SEED="${AUDIOFORGE_RANDOM_SEED:-42}"
OUTPUT_DIR="${AUDIOFORGE_RANDOM_OUTPUT_DIR:-outputs/random_subset_scratch}"

echo "[random] AudioForge random subset training started"
echo "[random] train samples: ${TRAIN_SAMPLES}"
echo "[random] val samples: ${VAL_SAMPLES}"
echo "[random] seed: ${SEED}"
echo "[random] output dir: ${OUTPUT_DIR}"

if [ ! -f "configs/fsd50k/random_subset.yaml" ]; then
  echo "[random][error] configs/fsd50k/random_subset.yaml not found"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/train.csv" ]; then
  echo "[random][error] full train manifest not found"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/val.csv" ]; then
  echo "[random][error] full val manifest not found"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/label_map.json" ]; then
  echo "[random][error] label_map.json not found"
  exit 1
fi

python - <<PY
import random
from pathlib import Path

from audioforge.data.manifests import read_manifest, write_manifest

train_samples = int("${TRAIN_SAMPLES}")
val_samples = int("${VAL_SAMPLES}")
seed = int("${SEED}")

random.seed(seed)

out_dir = Path("data/manifests/fsd50k/random_subset")
out_dir.mkdir(parents=True, exist_ok=True)

train_rows = read_manifest("data/manifests/fsd50k/train.csv")
val_rows = read_manifest("data/manifests/fsd50k/val.csv")

if train_samples > len(train_rows):
    raise SystemExit(f"Requested {train_samples} train samples, only {len(train_rows)} available")

if val_samples > len(val_rows):
    raise SystemExit(f"Requested {val_samples} val samples, only {len(val_rows)} available")

train_subset = random.sample(train_rows, train_samples)
val_subset = random.sample(val_rows, val_samples)

train_stats = write_manifest(out_dir / "train.csv", train_subset)
val_stats = write_manifest(out_dir / "val.csv", val_subset)

print("[random] wrote:", out_dir / "train.csv")
print("[random] wrote:", out_dir / "val.csv")
print("[random] train subset rows:", train_stats.rows)
print("[random] val subset rows:", val_stats.rows)
print("[random] train subset hours:", train_stats.total_duration_hours)
print("[random] val subset hours:", val_stats.total_duration_hours)
PY

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  MIXED_PRECISION="${AUDIOFORGE_MIXED_PRECISION:-fp16}"
  echo "[random] GPU detected. mixed_precision=${MIXED_PRECISION}"
else
  MIXED_PRECISION="no"
  echo "[random] no GPU detected. mixed_precision=no"
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

set +e
python -m audioforge.training.train_fsd50k \
  --config configs/fsd50k/random_subset.yaml \
  --mixed-precision "${MIXED_PRECISION}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train_stdout.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "[random][error] training failed"
  exit "${TRAIN_EXIT}"
fi

echo "[random] checking outputs"

test -f "${OUTPUT_DIR}/train_config.json"
test -f "${OUTPUT_DIR}/distributed.json"
test -f "${OUTPUT_DIR}/best/scratch_cnn_best.pt"
test -f "${OUTPUT_DIR}/best/best_metrics.json"

python - <<PY
import json
import re
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
log_path = output_dir / "train_stdout.log"
metrics_path = output_dir / "best" / "best_metrics.json"

text = log_path.read_text(encoding="utf-8", errors="replace")
losses = [float(match.group(1)) for match in re.finditer(r"loss=([0-9]+\\.[0-9]+)", text)]

if len(losses) < 2:
    raise SystemExit("[random][error] could not find enough logged losses")

first_window = losses[: min(5, len(losses))]
last_window = losses[-min(5, len(losses)) :]

first_avg = sum(first_window) / len(first_window)
last_avg = sum(last_window) / len(last_window)

print("[random] first logged losses:", first_window)
print("[random] last logged losses:", last_window)
print("[random] first avg loss:", first_avg)
print("[random] last avg loss:", last_avg)

if last_avg >= first_avg:
    raise SystemExit(
        f"[random][error] loss did not decrease: first_avg={first_avg:.6f}, last_avg={last_avg:.6f}"
    )

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
    raise SystemExit(f"[random][error] missing metric keys: {missing}")

if metrics["num_labels"] != 200:
    raise SystemExit(f"[random][error] expected 200 labels, got {metrics['num_labels']}")

print("[random] best mAP:", metrics["mAP"])
print("[random] micro F1:", metrics["micro_f1"])
print("[random] macro F1:", metrics["macro_f1"])
print("[random] num samples:", metrics["num_samples"])
print("[random] num labels:", metrics["num_labels"])
print("[random] metrics ok")
PY

echo "[random] generated files:"
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort

echo "[random] RANDOM SUBSET TRAINING PASSED ✅"
