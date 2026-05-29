#!/usr/bin/env bash
# Parallel feature cache builder for HF vision encoders.
set -euo pipefail

: "${MANIFEST:?Set MANIFEST to a Laguna Vision train/eval JSONL manifest.}"
: "${FEATURE_CACHE_DIR:?Set FEATURE_CACHE_DIR to a directory on the Prime persistent disk.}"

NPROC="${NPROC:-8}"
ENCODER="${ENCODER:-hf}"
VISION_TOWER="${VISION_TOWER:-google/siglip-so400m-patch14-384}"
MAX_TILES="${MAX_TILES:-1}"

mkdir -p "${FEATURE_CACHE_DIR}"

for rank in $(seq 0 "$((NPROC - 1))"); do
  (
    export CUDA_VISIBLE_DEVICES="${rank}"
    "${LAGUNA_VISION_CLI:-laguna-vision}" cache-visual-features \
      --manifest "${MANIFEST}" \
      --output-dir "${FEATURE_CACHE_DIR}" \
      --encoder "${ENCODER}" \
      --encoder-id "${VISION_TOWER}" \
      --max-tiles "${MAX_TILES}" \
      --device cuda \
      --shard-index "${rank}" \
      --num-shards "${NPROC}" \
      ${OVERWRITE_FEATURES:+--overwrite}
  ) &
done

wait
