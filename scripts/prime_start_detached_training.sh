#!/usr/bin/env bash
# Starts a detached training job on an already-created Prime pod.
set -euo pipefail

: "${PRIME_SSH_TARGET:?Set PRIME_SSH_TARGET, for example ubuntu@1.2.3.4.}"
: "${LAGUNA_VLM_ROOT:?Set LAGUNA_VLM_ROOT to the mounted Prime persistent disk path on the pod.}"

SSH_KEY="${SSH_KEY:-${HOME}/.ssh/prime_intellect}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-~/laguna-vision}"
MAX_RUNTIME="${MAX_RUNTIME:-8h}"
RUN_NAME="${RUN_NAME:-laguna-vision-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_HF_HOME="${HF_HOME:-${LAGUNA_VLM_ROOT}/hf_home}"
SSH_OPTS=(-i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new)

rsync -az \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude /.git \
  --exclude /.smoke \
  --exclude /data \
  --exclude /checkpoints \
  --exclude /outputs \
  --exclude /runs \
  --exclude /venv \
  ./ "${PRIME_SSH_TARGET}:${REMOTE_REPO_DIR}/"

ssh "${SSH_OPTS[@]}" "${PRIME_SSH_TARGET}" "mkdir -p '${REMOTE_HF_HOME}' && chmod 700 '${REMOTE_HF_HOME}'"
if [[ -n "${HF_TOKEN:-}" ]]; then
  printf '%s' "${HF_TOKEN}" | ssh "${SSH_OPTS[@]}" "${PRIME_SSH_TARGET}" "umask 077 && cat > '${REMOTE_HF_HOME}/token'"
elif [[ -f "${HF_TOKEN_FILE:-${HOME}/.cache/huggingface/token}" ]]; then
  rsync -az \
    -e "ssh ${SSH_OPTS[*]}" \
    "${HF_TOKEN_FILE:-${HOME}/.cache/huggingface/token}" \
    "${PRIME_SSH_TARGET}:${REMOTE_HF_HOME}/token"
  ssh "${SSH_OPTS[@]}" "${PRIME_SSH_TARGET}" "chmod 600 '${REMOTE_HF_HOME}/token'"
fi

ssh "${SSH_OPTS[@]}" "${PRIME_SSH_TARGET}" "bash -lc '
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -m pip --version >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update >/tmp/laguna_vision_apt_update.log
    apt-get install -y python3-pip python3-venv python3.10-venv >/tmp/laguna_vision_apt_install.log
  fi
  if [[ ! -x venv/bin/python || ! -f venv/bin/activate ]]; then
    rm -rf venv
    python3 -m venv venv
  fi
  source venv/bin/activate
  python -m pip install -U pip >/tmp/laguna_vision_pip_upgrade.log
  python -m pip install -e \".[llama,data,publish]\" >/tmp/laguna_vision_install.log
  mkdir -p \"${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}\"
  export LAGUNA_VLM_ROOT=\"${LAGUNA_VLM_ROOT}\"
  export MAX_RUNTIME=\"${MAX_RUNTIME}\"
  export RUN_NAME=\"${RUN_NAME}\"
  export TRAIN_COUNT=\"${TRAIN_COUNT:-30000}\"
  export EVAL_COUNT=\"${EVAL_COUNT:-1000}\"
  export MODEL_ID=\"${MODEL_ID:-poolside/Laguna-XS.2}\"
  export VISION_TOWER=\"${VISION_TOWER:-google/siglip-so400m-patch14-384}\"
  export MAX_TILES=\"${MAX_TILES:-1}\"
  export VISUAL_TOKENS=\"${VISUAL_TOKENS:-256}\"
  export BATCH_SIZE=\"${BATCH_SIZE:-1}\"
  export GRAD_ACCUM=\"${GRAD_ACCUM:-8}\"
  export SAVE_EVERY=\"${SAVE_EVERY:-50}\"
  export NPROC=\"${NPROC:-8}\"
  export DATA_DIR=\"${DATA_DIR:-${LAGUNA_VLM_ROOT}/datasets/hf_vqa}\"
  export FEATURE_CACHE_DIR=\"${FEATURE_CACHE_DIR:-${LAGUNA_VLM_ROOT}/feature_cache/siglip-so400m-patch14-384-tiles${MAX_TILES:-1}}\"
  export OUTPUT_DIR=\"${OUTPUT_DIR:-${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}}\"
  export INIT_CHECKPOINT=\"${INIT_CHECKPOINT:-}\"
  export HF_HOME=\"${REMOTE_HF_HOME}\"
  export PUBLISH_ON_EXIT=\"${PUBLISH_ON_EXIT:-1}\"
  export PUBLISH_DURING_RUN=\"${PUBLISH_DURING_RUN:-1}\"
  export HF_PUBLISH_INTERVAL=\"${HF_PUBLISH_INTERVAL:-300}\"
  export HF_REPO_ID=\"${HF_REPO_ID:-}\"
  export HF_PRIVATE=\"${HF_PRIVATE:-1}\"
  export HF_PATH_IN_REPO=\"${HF_PATH_IN_REPO:-${RUN_NAME}}\"
  export TERMINATE_ON_EXIT=\"${TERMINATE_ON_EXIT:-0}\"
  export PRIME_POD_ID=\"${PRIME_POD_ID:-}\"
  nohup bash scripts/prime_budget_training_job.sh > \"${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}/launcher.log\" 2>&1 < /dev/null &
  echo \$! > \"${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}/pid\"
  echo \"run_name=${RUN_NAME}\"
  echo \"pid=\$(cat \"${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}/pid\")\"
  echo \"log=${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}/job.log\"
'"
