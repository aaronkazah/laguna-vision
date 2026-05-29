#!/usr/bin/env bash
set -euo pipefail

: "${LAGUNA_VLM_ROOT:?Set LAGUNA_VLM_ROOT to the mounted Prime persistent disk path.}"

RUN_NAME="${RUN_NAME:-laguna-vision-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_RUNTIME="${MAX_RUNTIME:-8h}"
TRAIN_COUNT="${TRAIN_COUNT:-30000}"
EVAL_COUNT="${EVAL_COUNT:-1000}"
MODEL_ID="${MODEL_ID:-poolside/Laguna-XS.2}"
VISION_TOWER="${VISION_TOWER:-google/siglip-so400m-patch14-384}"
MAX_TILES="${MAX_TILES:-1}"
VISUAL_TOKENS="${VISUAL_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAVE_EVERY="${SAVE_EVERY:-50}"
MAX_ITEMS="${MAX_ITEMS:-0}"
NPROC="${NPROC:-8}"
DATA_DIR="${DATA_DIR:-${LAGUNA_VLM_ROOT}/datasets/hf_vqa}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATA_DIR}/train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${DATA_DIR}/eval.jsonl}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${LAGUNA_VLM_ROOT}/feature_cache/siglip-so400m-patch14-384-tiles${MAX_TILES}}"
OUTPUT_DIR="${OUTPUT_DIR:-${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}}"
CLI="${LAGUNA_VISION_CLI:-laguna-vision}"
HF_HOME="${HF_HOME:-${LAGUNA_VLM_ROOT}/hf_home}"
PUBLISH_ON_EXIT="${PUBLISH_ON_EXIT:-1}"
PUBLISH_DURING_RUN="${PUBLISH_DURING_RUN:-1}"
HF_PUBLISH_INTERVAL="${HF_PUBLISH_INTERVAL:-300}"
export BATCH_SIZE CLI DATA_DIR EVAL_COUNT EVAL_MANIFEST FEATURE_CACHE_DIR GRAD_ACCUM HF_HOME INIT_CHECKPOINT
export MAX_ITEMS MAX_TILES MODEL_ID NPROC OUTPUT_DIR RUN_NAME SAVE_EVERY TRAIN_COUNT TRAIN_MANIFEST VISION_TOWER VISUAL_TOKENS

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${FEATURE_CACHE_DIR}"
exec > >(tee -a "${LOG_DIR}/job.log") 2>&1

echo "run_name=${RUN_NAME}"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "max_runtime=${MAX_RUNTIME}"
echo "train_count=${TRAIN_COUNT}"
echo "eval_count=${EVAL_COUNT}"
echo "model_id=${MODEL_ID}"
echo "vision_tower=${VISION_TOWER}"
echo "max_tiles=${MAX_TILES}"
echo "visual_tokens=${VISUAL_TOKENS}"
echo "max_items=${MAX_ITEMS}"
echo "output_dir=${OUTPUT_DIR}"
echo "feature_cache_dir=${FEATURE_CACHE_DIR}"
echo "hf_home=${HF_HOME}"
echo "init_checkpoint=${INIT_CHECKPOINT:-}"
echo "publish_during_run=${PUBLISH_DURING_RUN}"

publish_checkpoint() {
  local checkpoint_dir="$1"
  [[ "${HF_REPO_ID:-}" != "" ]] || return 0
  [[ -f "${checkpoint_dir}/projector.pt" ]] || return 0
  [[ -f "${checkpoint_dir}/projector_spec.json" ]] || return 0
  [[ -f "${checkpoint_dir}/train_report.json" ]] || return 0

  CHECKPOINT_DIR="${checkpoint_dir}" \
    HF_PRIVATE="${HF_PRIVATE:-1}" \
    PATH_IN_REPO="${HF_PATH_IN_REPO:-${RUN_NAME}}/$(basename "${checkpoint_dir}")" \
    scripts/publish_hf_checkpoint.sh
}

publish_new_checkpoints() {
  local watched_pid="$1"
  local published_dir="${LOG_DIR}/published_hf"
  mkdir -p "${published_dir}"

  while kill -0 "${watched_pid}" >/dev/null 2>&1; do
    for checkpoint_dir in "${OUTPUT_DIR}"/step_*; do
      [[ -d "${checkpoint_dir}" ]] || continue
      marker="${published_dir}/$(basename "${checkpoint_dir}")"
      [[ ! -f "${marker}" ]] || continue
      publish_checkpoint "${checkpoint_dir}" && touch "${marker}"
    done
    sleep "${HF_PUBLISH_INTERVAL}"
  done

  for checkpoint_dir in "${OUTPUT_DIR}"/step_*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    marker="${published_dir}/$(basename "${checkpoint_dir}")"
    [[ ! -f "${marker}" ]] || continue
    publish_checkpoint "${checkpoint_dir}" && touch "${marker}"
  done
}

publish_latest_checkpoint() {
  [[ "${PUBLISH_ON_EXIT}" == "1" ]] || return 0

  latest="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort | tail -1 || true)"
  if [[ -z "${latest}" && -f "${OUTPUT_DIR}/projector.pt" ]]; then
    latest="${OUTPUT_DIR}"
  fi
  [[ -n "${latest}" ]] || return 0

  publish_checkpoint "${latest}" || true
}

terminate_on_exit() {
  status=$?
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_status=${status}"
  publish_latest_checkpoint
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

training_script="${LOG_DIR}/run_training.sh"
cat > "${training_script}" <<'TRAINING_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

line_count() {
  local path="$1"
  [[ -f "${path}" ]] || {
    echo 0
    return
  }
  wc -l < "${path}" | tr -d ' '
}

train_rows="$(line_count "${TRAIN_MANIFEST}")"
eval_rows="$(line_count "${EVAL_MANIFEST}")"
if (( train_rows < TRAIN_COUNT || eval_rows < EVAL_COUNT )); then
  "${CLI}" hf-materialize \
    --output-dir "${DATA_DIR}" \
    --train-count "${TRAIN_COUNT}" \
    --eval-count "${EVAL_COUNT}"
else
  echo "using_cached_dataset train_rows=${train_rows} eval_rows=${eval_rows}"
fi

export MANIFEST="${TRAIN_MANIFEST}"
export FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}"
export VISION_TOWER="${VISION_TOWER}"
export MAX_TILES="${MAX_TILES}"
export NPROC="${NPROC}"
scripts/laguna_llava_cache_features.sh

train_args=(
    --backbone laguna \
    --model-id "${MODEL_ID}" \
    --manifest "${TRAIN_MANIFEST}" \
    --eval-manifest "${EVAL_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --feature-cache-dir "${FEATURE_CACHE_DIR}" \
    --encoder hf \
    --encoder-id "${VISION_TOWER}" \
    --projector resampler \
    --visual-tokens "${VISUAL_TOKENS}" \
    --max-tiles "${MAX_TILES}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --epochs 1 \
    --lr 2e-5 \
    --warmup-ratio 0.03 \
    --save-every "${SAVE_EVERY}" \
    --lora-rank 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --device cuda \
    --vision-device cuda
)

if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  train_args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi
if (( MAX_ITEMS > 0 )); then
  train_args+=(--max-items "${MAX_ITEMS}")
fi

NPROC="${NPROC}" scripts/train_visual_bridge_ddp.sh "${train_args[@]}"

if [[ -f "${OUTPUT_DIR}/projector.pt" ]]; then
  "${CLI}" eval-ablation \
    --manifest "${EVAL_MANIFEST}" \
    --checkpoint "${OUTPUT_DIR}/projector.pt" \
    --output "${OUTPUT_DIR}/ablation.jsonl" \
    --backbone laguna \
    --model-id "${MODEL_ID}" \
    --threshold 0.15 \
    --device cuda \
    --vision-device cuda
fi
TRAINING_SCRIPT
chmod +x "${training_script}"

timeout "${MAX_RUNTIME}" bash "${training_script}" &
run_pid=$!
publisher_pid=""
if [[ "${PUBLISH_DURING_RUN}" == "1" && -n "${HF_REPO_ID:-}" ]]; then
  publish_new_checkpoints "${run_pid}" &
  publisher_pid=$!
fi

set +e
wait "${run_pid}"
run_status=$?
if [[ -n "${publisher_pid}" ]]; then
  wait "${publisher_pid}" || true
fi
set -e
exit "${run_status}"
