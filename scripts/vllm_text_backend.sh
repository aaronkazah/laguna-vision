#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible backend for the Laguna text model.
set -euo pipefail

: "${VLLM_MODEL:?Set VLLM_MODEL to the merged Laguna model path or base model id.}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-laguna-vision}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

args=(
  serve "${VLLM_MODEL}"
  --runner generate
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --served-model-name "${VLLM_SERVED_MODEL_NAME}"
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
  --trust-remote-code
  --enable-prompt-embeds
)

if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  args+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
fi

if [[ -n "${VLLM_DTYPE:-}" ]]; then
  args+=(--dtype "${VLLM_DTYPE}")
fi

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  args+=(--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
fi

if [[ -n "${VLLM_LORA_DIR:-}" ]]; then
  VLLM_LORA_NAME="${VLLM_LORA_NAME:-laguna-vision}"
  args+=(--enable-lora --lora-modules "${VLLM_LORA_NAME}=${VLLM_LORA_DIR}")
fi

exec vllm "${args[@]}" "$@"
