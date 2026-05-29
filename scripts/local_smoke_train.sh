#!/usr/bin/env bash
# Local end-to-end check for data, training, inference, and evaluation.
set -euo pipefail

ROOT="${ROOT:-.smoke/laguna-vision}"
MODEL_ID="${MODEL_ID:-sshleifer/tiny-gpt2}"
CLI="${LAGUNA_VISION_CLI:-laguna-vision}"

mkdir -p "${ROOT}"

"${CLI}" scene-dataset \
  --output-dir "${ROOT}/data" \
  --train-count "${TRAIN_COUNT:-8}" \
  --eval-count "${EVAL_COUNT:-2}"

"${CLI}" train-visual-bridge \
  --backbone hf \
  --model-id "${MODEL_ID}" \
  --manifest "${ROOT}/data/train.jsonl" \
  --eval-manifest "${ROOT}/data/eval.jsonl" \
  --output-dir "${ROOT}/checkpoints/tiny" \
  --epochs 1 \
  --max-items "${MAX_ITEMS:-8}" \
  --encoder pil \
  --projector resampler \
  --visual-tokens 4 \
  --batch-size 2 \
  --grad-accum 1 \
  --save-every 2 \
  --device cpu \
  --vision-device cpu

first_image="$(
  python - "${ROOT}/data/eval.jsonl" <<'PY'
import json
import sys
from pathlib import Path

row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[0])
print(Path(sys.argv[1]).parent / row["image"])
PY
)"

"${CLI}" ask-image \
  --checkpoint "${ROOT}/checkpoints/tiny/projector.pt" \
  --image "${first_image}" \
  --question "Describe the image in one short sentence." \
  --backbone hf \
  --model-id "${MODEL_ID}" \
  --device cpu \
  --vision-device cpu

"${CLI}" eval-ablation \
  --manifest "${ROOT}/data/eval.jsonl" \
  --checkpoint "${ROOT}/checkpoints/tiny/projector.pt" \
  --output "${ROOT}/eval/ablation.jsonl" \
  --backbone hf \
  --model-id "${MODEL_ID}" \
  --limit 2 \
  --threshold 0.0 \
  --device cpu \
  --vision-device cpu
