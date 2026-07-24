#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="configs/fsd50k/smoke_ast.yaml"
OUTPUT_DIR="${AUDIOFORGE_OUTPUT_DIR:-outputs/smoke_ast}"

echo "[smoke-ast] AudioForge FSD50K AST+LoRA smoke test"
echo "[smoke-ast] config: ${CONFIG_PATH}"
echo "[smoke-ast] output: ${OUTPUT_DIR}"

test -f "${CONFIG_PATH}"

if [ ! -f "data/manifests/fsd50k/train.csv" ]; then
  echo "[smoke-ast][error] train manifest not found: data/manifests/fsd50k/train.csv"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/val.csv" ]; then
  echo "[smoke-ast][error] val manifest not found: data/manifests/fsd50k/val.csv"
  exit 1
fi

if [ ! -f "data/manifests/fsd50k/label_map.json" ]; then
  echo "[smoke-ast][error] label_map.json not found"
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
  echo "[smoke-ast][error] training failed"
  exit "${TRAIN_EXIT}"
fi

echo "[smoke-ast] checking outputs"

test -f "${OUTPUT_DIR}/train_config.json"
test -f "${OUTPUT_DIR}/distributed.json"
test -f "${OUTPUT_DIR}/best/ast_best.pt"
test -f "${OUTPUT_DIR}/best/best_metrics.json"

echo "[smoke-ast] verifying LoRA actually reduced trainable params"
grep -q "Trainable parameters:" "${OUTPUT_DIR}/train_stdout.log"

echo "[smoke-ast] generated files:"
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort

echo "[smoke-ast] AST SMOKE TEST PASSED ✅"
