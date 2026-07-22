#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export AUDIOFORGE_NUM_PROCESSES="${AUDIOFORGE_NUM_PROCESSES:-2}"
exec bash scripts/train_scratch_cnn.sh
