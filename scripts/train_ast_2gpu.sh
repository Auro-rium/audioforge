#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="${AUDIOFORGE_CONFIG:-configs/fsd50k/ast_2gpu.yaml}"
OUTPUT_DIR="${AUDIOFORGE_OUTPUT_DIR:-outputs/fsd50k/ast_2gpu}"
NUM_PROCESSES="${AUDIOFORGE_NUM_PROCESSES:-1}"
MIXED_PRECISION="${AUDIOFORGE_MIXED_PRECISION:-fp16}"

echo "[ast] AudioForge AST fine-tune (num_processes=${NUM_PROCESSES})"
echo "[ast] config: ${CONFIG_PATH}"
echo "[ast] output: ${OUTPUT_DIR}"
echo "[ast] num_processes: ${NUM_PROCESSES}"
echo "[ast] mixed_precision: ${MIXED_PRECISION}"

test -f "${CONFIG_PATH}"
test -f data/manifests/fsd50k/train.csv
test -f data/manifests/fsd50k/val.csv
test -f data/manifests/fsd50k/label_map.json

python - <<'PY'
import json
from audioforge.data.manifests import read_manifest

train = read_manifest("data/manifests/fsd50k/train.csv")
val = read_manifest("data/manifests/fsd50k/val.csv")

with open("data/manifests/fsd50k/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

print("[ast] train rows:", len(train))
print("[ast] val rows:", len(val))
print("[ast] labels:", label_map["num_labels"])
assert len(train) == 36796
assert len(val) == 4170
assert label_map["num_labels"] == 200
PY

python - <<'PY'
from audioforge.training.trainer import load_train_config
from audioforge.models.ast import create_ast_classifier
from transformers import AutoFeatureExtractor

cfg = load_train_config("configs/fsd50k/ast_2gpu.yaml")
print("[ast] loading:", cfg.pretrained_name_or_path)
print("[ast] use_lora:", cfg.use_lora, "freeze_backbone:", cfg.freeze_backbone)

_ = AutoFeatureExtractor.from_pretrained(cfg.pretrained_name_or_path)
model = create_ast_classifier(
    pretrained_name_or_path=cfg.pretrained_name_or_path,
    num_labels=cfg.num_labels,
    dropout=cfg.dropout,
    freeze_backbone=cfg.freeze_backbone,
    use_lora=cfg.use_lora,
    lora_r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    lora_target_modules=cfg.lora_target_modules,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print("[ast] model load ok:", model.__class__.__name__)
print(f"[ast] trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  echo "[ast] detected GPUs: ${GPU_COUNT}"
  nvidia-smi
else
  GPU_COUNT="0"
  echo "[ast][warning] nvidia-smi not found"
fi

if [ "${GPU_COUNT}" -lt "${NUM_PROCESSES}" ]; then
  echo "[ast][error] requested ${NUM_PROCESSES} processes but only ${GPU_COUNT} GPUs detected"
  echo "[ast][hint] do not run full AST on CPU."
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
  echo "[ast][error] training failed"
  exit "${TRAIN_EXIT}"
fi

test -f "${OUTPUT_DIR}/train_config.json"
test -f "${OUTPUT_DIR}/distributed.json"
test -f "${OUTPUT_DIR}/best/ast_best.pt"
test -f "${OUTPUT_DIR}/best/best_metrics.json"

python scripts/make_fsd50k_benchmark_row.py \
  --run-dir "${OUTPUT_DIR}" \
  --model-name ast \
  --out reports/metrics/ast_benchmark_row.json \
  --markdown-out reports/fsd50k_ast_row.md

cat reports/fsd50k_ast_row.md

echo "[ast] AST FINE-TUNE PASSED ✅"
