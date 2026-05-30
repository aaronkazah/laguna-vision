#!/usr/bin/env bash
# LLaVA-compatible visual instruction tuning for Laguna.
set -euo pipefail

: "${LAGUNA_VLM_ROOT:?Set LAGUNA_VLM_ROOT to a persistent run root, for example ${PWD}/runs/laguna-vlm.}"

ALLOW_LOCAL_OUTPUT="${ALLOW_LOCAL_OUTPUT:-${ALLOW_POD_LOCAL_OUTPUT:-0}}"
if [[ "${ALLOW_LOCAL_OUTPUT}" != "1" && "${LAGUNA_VLM_ROOT}" != /mnt/* && "${LAGUNA_VLM_ROOT}" != /workspace/* && "${LAGUNA_VLM_ROOT}" != /prime/* && "${LAGUNA_VLM_ROOT}" != /data/* ]]; then
  echo "Refusing to write production checkpoints outside a mounted persistent path: ${LAGUNA_VLM_ROOT}" >&2
  echo "Set ALLOW_LOCAL_OUTPUT=1 for local reproduction runs." >&2
  exit 2
fi

MODEL_ID="${MODEL_ID:-poolside/Laguna-XS.2}"
VISION_TOWER="${VISION_TOWER:-google/siglip-so400m-patch14-384}"
MANIFEST="${MANIFEST:-${LAGUNA_VLM_ROOT}/manifests/llava_instruction/train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${LAGUNA_VLM_ROOT}/manifests/llava_instruction/eval.jsonl}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${LAGUNA_VLM_ROOT}/checkpoints/laguna_stage1_alignment/projector.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${LAGUNA_VLM_ROOT}/checkpoints/laguna_stage2_instruction}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${LAGUNA_VLM_ROOT}/feature_cache/llava_stage2}"
NPROC="${NPROC:-8}"
SAVE_EVERY="${SAVE_EVERY:-250}"

eval_args=()
if [[ -f "${EVAL_MANIFEST}" ]]; then
  eval_args=(--eval-manifest "${EVAL_MANIFEST}")
fi
max_items_args=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
  max_items_args=(--max-items "${MAX_ITEMS}")
fi
init_args=(--init-checkpoint "${INIT_CHECKPOINT:-${STAGE1_CHECKPOINT}}")
if [[ -n "${INIT_LORA_DIR:-}" ]]; then
  init_args+=(--init-lora-dir "${INIT_LORA_DIR}")
fi

NPROC="${NPROC}" "$(dirname "$0")/train_visual_bridge_ddp.sh" \
  --backbone laguna \
  --model-id "${MODEL_ID}" \
  --manifest "${MANIFEST}" \
  "${eval_args[@]}" \
  "${max_items_args[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --feature-cache-dir "${FEATURE_CACHE_DIR}" \
  "${init_args[@]}" \
  --encoder hf \
  --encoder-id "${VISION_TOWER}" \
  --projector resampler \
  --visual-tokens "${VISUAL_TOKENS:-256}" \
  --max-tiles "${MAX_TILES:-1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-16}" \
  --epochs "${EPOCHS:-1}" \
  --lr "${LR:-2e-5}" \
  --warmup-ratio "${WARMUP_RATIO:-0.03}" \
  --save-every "${SAVE_EVERY}" \
  --lora-rank "${LORA_RANK:-64}" \
  --lora-alpha "${LORA_ALPHA:-128}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --device cuda \
  --vision-device cuda
