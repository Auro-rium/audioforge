#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="configs/fsd50k/smoke.yaml"
OUTPUT_DIR="${AUDIOFORGE_OUTPUT_DIR:-outputs/smoke_scratch}"

echo "[smoke] AudioForge FSD50K smoke test"
echo "[smoke] config: ${CONFIG_PATH}"
echo "[smoke] output: ${OUTPUT_DIR}"

test -f "${CONFIG_PATH}"

if [ ! -f "data/manifests/fsd50k/train.csv" ]; then
  echo "[smoke][error] train manifest not found: data/manifests/fsd50k/train.csv"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/val.csv" ]; then
  echo "[smoke][error] val manifest not found: data/manifests/fsd50k/val.csv"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/label_map.json" ]; then
  echo "[smoke][error] label_map.json not found"
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

set +e
python -m audioforge.training.train_fsd50k \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train_stdout.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "[smoke][error] training failed"
  exit "${TRAIN_EXIT}"
fi

echo "[smoke] checking outputs"

test -f "${OUTPUT_DIR}/train_config.json"
test -f "${OUTPUT_DIR}/distributed.json"
test -f "${OUTPUT_DIR}/best/scratch_cnn_best.pt"
test -f "${OUTPUT_DIR}/best/best_metrics.json"

echo "[smoke] generated files:"
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort

echo "[smoke] SMOKE TEST PASSED ✅"
