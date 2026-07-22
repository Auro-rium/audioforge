#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="${AUDIOFORGE_DCASE_CONFIG:-configs/dcase/ensemble.yaml}"
MANIFEST_PATH="${AUDIOFORGE_DCASE_MANIFEST:-data/manifests/dcase2024/all.csv}"
OUTPUT_BASE="${AUDIOFORGE_DCASE_OUTPUT:-outputs/dcase/ensemble}"

test -f "${CONFIG_PATH}"
test -f "${MANIFEST_PATH}"

python -m audioforge.training.train_dcase \
  --config "${CONFIG_PATH}" \
  --manifest "${MANIFEST_PATH}" \
  --output "${OUTPUT_BASE}"

PREDICTIONS="${OUTPUT_BASE}_predictions.csv"
METRICS="${OUTPUT_BASE}_metrics.json"

python - "${PREDICTIONS}" "${METRICS}" <<'PY'
import json
import sys
from audioforge.evaluation.dcase_metrics import compute_dcase_metrics, metrics_to_dict, read_dcase_predictions

predictions = read_dcase_predictions(sys.argv[1])
metrics = compute_dcase_metrics(predictions)
with open(sys.argv[2], "w", encoding="utf-8") as file:
    json.dump(metrics_to_dict(metrics), file, indent=2)
print(json.dumps(metrics_to_dict(metrics), indent=2))
PY
