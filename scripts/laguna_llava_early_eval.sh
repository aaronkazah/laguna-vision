#!/usr/bin/env bash
# Evaluate an early Laguna VLM checkpoint on a real image and an optional held-out
# manifest. This is meant to run during stage-2 checkpointing.
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to a projector.pt on the Prime persistent disk.}"
: "${IMAGE:?Set IMAGE to a held-out real image path.}"

QUESTION="${QUESTION:-Explain this image in detail.}"
DEVICE="${DEVICE:-cuda}"
VISION_DEVICE="${VISION_DEVICE:-cuda}"

"${LAGUNA_VISION_CLI:-laguna-vision}" ask-image \
  --checkpoint "${CHECKPOINT}" \
  --image "${IMAGE}" \
  --question "${QUESTION}" \
  --device "${DEVICE}" \
  --vision-device "${VISION_DEVICE}"

if [[ -n "${MANIFEST:-}" ]]; then
  OUTPUT="${OUTPUT:-$(dirname "${CHECKPOINT}")/early_ablation.jsonl}"
  "${LAGUNA_VISION_CLI:-laguna-vision}" eval-ablation \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --output "${OUTPUT}" \
    --limit "${LIMIT:-64}" \
    --threshold "${THRESHOLD:-0.15}" \
    --device "${DEVICE}" \
    --vision-device "${VISION_DEVICE}"
fi
