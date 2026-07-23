#!/usr/bin/env bash
set -euo pipefail

# Downloads and assembles the raw FSD50K corpus into the layout
# audioforge/data/fsd50k.py expects:
#
#   data/raw/fsd50k/FSD50K.dev_audio/*.wav
#   data/raw/fsd50k/FSD50K.eval_audio/*.wav
#   data/raw/fsd50k/FSD50K.ground_truth/{dev.csv,eval.csv,vocabulary.csv}
#
# Run this ON the training instance, not your laptop (see conversation:
# inbound transfer to EC2 is free, and it avoids a slow local re-upload).
#
# File list and URL scheme verified directly against the Zenodo API record
# (https://zenodo.org/api/records/4060432) on 2026-07-23: all 9 file keys
# below match the record's "files" array exactly (dev_audio.z01-z05+zip =
# 6 parts, eval_audio.z01+zip = 2 parts, ground_truth.zip = 1 part, total
# ~29GB compressed), and the download URL pattern was confirmed live with a
# HEAD request (200 OK, content-length matched the API's reported size).

cd "$(dirname "$0")/.."

RAW_ROOT="data/raw/fsd50k"
DOWNLOAD_DIR="${RAW_ROOT}/_download"
ZENODO_RECORD="https://zenodo.org/api/records/4060432/files"

FILES=(
  "FSD50K.dev_audio.z01"
  "FSD50K.dev_audio.z02"
  "FSD50K.dev_audio.z03"
  "FSD50K.dev_audio.z04"
  "FSD50K.dev_audio.z05"
  "FSD50K.dev_audio.zip"
  "FSD50K.eval_audio.z01"
  "FSD50K.eval_audio.zip"
  "FSD50K.ground_truth.zip"
)
# -----------------------------------------------------------------

mkdir -p "${DOWNLOAD_DIR}"

echo "[fsd50k] downloading ${#FILES[@]} files into ${DOWNLOAD_DIR}"

for name in "${FILES[@]}"; do
  dest="${DOWNLOAD_DIR}/${name}"
  if [ -f "${dest}" ]; then
    echo "[fsd50k] already present, skipping: ${name}"
    continue
  fi
  echo "[fsd50k] downloading ${name}"
  # -C - resumes a partial download instead of restarting from zero,
  # which matters a lot on a multi-GB file over a real network.
  curl -L -C - --fail --retry 5 --retry-delay 5 \
    -o "${dest}.part" \
    "${ZENODO_RECORD}/${name}/content"
  mv "${dest}.part" "${dest}"
done

echo "[fsd50k] all downloads complete"

reassemble_and_extract() {
  local final_part="$1"   # e.g. FSD50K.dev_audio.zip (the LAST part of the split set)
  local label="$2"

  echo "[fsd50k] reassembling split zip: ${label}"
  # zip -s 0 merges all .zNN parts sitting next to the given .zip into one
  # combined archive. All parts must be present in DOWNLOAD_DIR.
  ( cd "${DOWNLOAD_DIR}" && zip -q -s 0 "${final_part}" --out "combined_${final_part}" )

  echo "[fsd50k] extracting: ${label}"
  unzip -q "${DOWNLOAD_DIR}/combined_${final_part}" -d "${RAW_ROOT}"
  rm -f "${DOWNLOAD_DIR}/combined_${final_part}"
}

reassemble_and_extract "FSD50K.dev_audio.zip" "dev_audio"
reassemble_and_extract "FSD50K.eval_audio.zip" "eval_audio"

echo "[fsd50k] extracting ground truth (single-part zip, no reassembly needed)"
unzip -q "${DOWNLOAD_DIR}/FSD50K.ground_truth.zip" -d "${RAW_ROOT}"

echo "[fsd50k] sanity check"
DEV_COUNT=$(find "${RAW_ROOT}/FSD50K.dev_audio" -name '*.wav' | wc -l)
EVAL_COUNT=$(find "${RAW_ROOT}/FSD50K.eval_audio" -name '*.wav' | wc -l)
echo "[fsd50k] dev_audio wav files: ${DEV_COUNT}"
echo "[fsd50k] eval_audio wav files: ${EVAL_COUNT}"
test -f "${RAW_ROOT}/FSD50K.ground_truth/dev.csv"
test -f "${RAW_ROOT}/FSD50K.ground_truth/eval.csv"
test -f "${RAW_ROOT}/FSD50K.ground_truth/vocabulary.csv"
echo "[fsd50k] ground truth CSVs present"

read -r -p "[fsd50k] delete downloaded zip parts in ${DOWNLOAD_DIR} to free disk space? [y/N] " reply
if [[ "${reply}" =~ ^[Yy]$ ]]; then
  rm -rf "${DOWNLOAD_DIR}"
  echo "[fsd50k] removed ${DOWNLOAD_DIR}"
else
  echo "[fsd50k] keeping ${DOWNLOAD_DIR} (remove manually later if disk space is tight)"
fi

echo "[fsd50k] done. Next: python scripts/prepare_fsd50k.py --root ${RAW_ROOT}"
