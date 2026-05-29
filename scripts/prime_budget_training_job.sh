#!/usr/bin/env bash
# Runs on a Prime pod. It assumes the repo is present, dependencies are installed,
# and LAGUNA_VLM_ROOT points at a mounted persistent disk.
set -euo pipefail

: "${LAGUNA_VLM_ROOT:?Set LAGUNA_VLM_ROOT to the mounted Prime persistent disk path.}"

RUN_NAME="${RUN_NAME:-laguna-vlm-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_RUNTIME="${MAX_RUNTIME:-9h}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${LAGUNA_VLM_ROOT}/data/llava/train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${LAGUNA_VLM_ROOT}/data/llava/eval.jsonl}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${LAGUNA_VLM_ROOT}/feature_cache/clip-vit-large-patch14-336}"
LOG_DIR="${LOG_DIR:-${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}}"
CLI="${LAGUNA_VISION_CLI:-laguna-vision}"

mkdir -p "${LOG_DIR}" "${LAGUNA_VLM_ROOT}/checkpoints"
exec > >(tee -a "${LOG_DIR}/job.log") 2>&1

echo "run_name=${RUN_NAME}"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "max_runtime=${MAX_RUNTIME}"
echo "train_manifest=${TRAIN_MANIFEST}"
echo "eval_manifest=${EVAL_MANIFEST}"
echo "feature_cache_dir=${FEATURE_CACHE_DIR}"

terminate_on_exit() {
  status=$?
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_status=${status}"
  if [[ "${HF_REPO_ID:-}" != "" && "${PUBLISH_ON_EXIT:-0}" == "1" ]]; then
    latest="$(find "${LAGUNA_VLM_ROOT}/checkpoints/laguna_stage2_instruction" -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "${latest}" ]]; then
      CHECKPOINT_DIR="${latest}" HF_PRIVATE="${HF_PRIVATE:-1}" scripts/publish_hf_checkpoint.sh || true
    fi
  fi
  if [[ "${TERMINATE_ON_EXIT:-0}" == "1" ]]; then
    if [[ -z "${PRIME_POD_ID:-}" ]]; then
      echo "TERMINATE_ON_EXIT=1 but PRIME_POD_ID is unset; leaving pod running."
    elif command -v prime >/dev/null 2>&1; then
      prime pods terminate "${PRIME_POD_ID}" -y || true
    else
      echo "TERMINATE_ON_EXIT=1 but prime CLI is not installed in the pod; leaving pod running."
    fi
  fi
  exit "${status}"
}
trap terminate_on_exit EXIT

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "Missing ${TRAIN_MANIFEST}. Materialize or copy LLaVA-format data onto the persistent disk first." >&2
  exit 2
fi

timeout "${MAX_RUNTIME}" bash -lc '
  set -euo pipefail
  export MANIFEST="'"${TRAIN_MANIFEST}"'"
  export EVAL_MANIFEST="'"${EVAL_MANIFEST}"'"
  export FEATURE_CACHE_DIR="'"${FEATURE_CACHE_DIR}"'"

  scripts/laguna_llava_cache_features.sh
  MAX_ITEMS="'"${STAGE1_MAX_ITEMS:-300000}"'" scripts/laguna_llava_stage1.sh
  MAX_ITEMS="'"${STAGE2_MAX_ITEMS:-150000}"'" scripts/laguna_llava_stage2.sh
'

