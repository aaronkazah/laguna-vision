#!/usr/bin/env bash
# Start the Laguna Vision gateway that converts images into vLLM prompt_embeds.
set -euo pipefail

: "${LAGUNA_CHECKPOINT:?Set LAGUNA_CHECKPOINT to a checkpoint dir or projector.pt path.}"

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL="${VLLM_MODEL:-laguna-vision}"
LAGUNA_VLLM_HOST="${LAGUNA_VLLM_HOST:-0.0.0.0}"
LAGUNA_VLLM_PORT="${LAGUNA_VLLM_PORT:-8080}"

args=(
  serve
  --checkpoint "${LAGUNA_CHECKPOINT}"
  --vllm-base-url "${VLLM_BASE_URL}"
  --model "${VLLM_MODEL}"
  --host "${LAGUNA_VLLM_HOST}"
  --port "${LAGUNA_VLLM_PORT}"
)

if [[ -n "${LAGUNA_VISION_DEVICE:-}" ]]; then
  args+=(--vision-device "${LAGUNA_VISION_DEVICE}")
fi

if [[ -n "${LAGUNA_MAX_NEW_TOKENS:-}" ]]; then
  args+=(--max-new-tokens "${LAGUNA_MAX_NEW_TOKENS}")
fi

if [[ "${LAGUNA_VLLM_ALLOW_LOCAL_FILES:-0}" == "1" ]]; then
  args+=(--allow-local-files)
fi

exec laguna-vision-vllm "${args[@]}" "$@"
