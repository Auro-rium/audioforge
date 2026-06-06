#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="${AUDIOFORGE_CONFIG:-configs/fsd50k/scratch_cnn.yaml}"
OUTPUT_DIR="${AUDIOFORGE_OUTPUT_DIR:-outputs/fsd50k/scratch_cnn_full}"
NUM_PROCESSES="${AUDIOFORGE_NUM_PROCESSES:-2}"
MIXED_PRECISION="${AUDIOFORGE_MIXED_PRECISION:-fp16}"

echo "[scratch] AudioForge full Scratch CNN baseline"
echo "[scratch] config: ${CONFIG_PATH}"
echo "[scratch] output: ${OUTPUT_DIR}"
echo "[scratch] num_processes: ${NUM_PROCESSES}"
echo "[scratch] mixed_precision: ${MIXED_PRECISION}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[scratch][error] config not found: ${CONFIG_PATH}"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/train.csv" ]; then
  echo "[scratch][error] train manifest not found"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/val.csv" ]; then
  echo "[scratch][error] val manifest not found"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/label_map.json" ]; then
  echo "[scratch][error] label_map.json not found"
  exit 1
fi

python - <<'PY'
import json
from audioforge.data.manifests import read_manifest

train = read_manifest("data/manifests/fsd50k/train.csv")
val = read_manifest("data/manifests/fsd50k/val.csv")

with open("data/manifests/fsd50k/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

print("[scratch] train rows:", len(train))
print("[scratch] val rows:", len(val))
print("[scratch] labels:", label_map["num_labels"])
print("[scratch] train hours:", round(sum(r.duration for r in train) / 3600, 2))
print("[scratch] val hours:", round(sum(r.duration for r in val) / 3600, 2))

assert len(train) == 36796
assert len(val) == 4170
assert label_map["num_labels"] == 200
PY

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[scratch][error] accelerate command not found"
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  echo "[scratch] detected GPUs: ${GPU_COUNT}"
  nvidia-smi
else
  GPU_COUNT="0"
  echo "[scratch][warning] nvidia-smi not found"
fi

if [ "${GPU_COUNT}" -lt "${NUM_PROCESSES}" ]; then
  echo "[scratch][error] requested ${NUM_PROCESSES} processes but only ${GPU_COUNT} GPUs detected"
  echo "[scratch][hint] for CPU/debug only, run: AUDIOFORGE_NUM_PROCESSES=1 AUDIOFORGE_MIXED_PRECISION=no bash scripts/train_scratch_cnn.sh"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

set +e
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  -m audioforge.training.train_fsd50k \
  --config "${CONFIG_PATH}" \
  --mixed-precision "${MIXED_PRECISION}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train_stdout.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "[scratch][error] training failed"
  exit "${TRAIN_EXIT}"
fi

echo "[scratch] checking outputs"

test -f "${OUTPUT_DIR}/train_config.json"
test -f "${OUTPUT_DIR}/distributed.json"
test -f "${OUTPUT_DIR}/best/scratch_cnn_best.pt"
test -f "${OUTPUT_DIR}/best/best_metrics.json"

python scripts/make_scratch_benchmark_row.py \
  --run-dir "${OUTPUT_DIR}" \
  --out reports/metrics/scratch_cnn_benchmark_row.json \
  --markdown-out reports/fsd50k_scratch_cnn_row.md

echo "[scratch] generated benchmark row:"
cat reports/fsd50k_scratch_cnn_row.md

echo "[scratch] FULL SCRATCH CNN BASELINE PASSED ✅"
