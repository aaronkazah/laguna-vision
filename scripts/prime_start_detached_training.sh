#!/usr/bin/env bash
# Starts a budget-capped training job on an already-created Prime pod. The job
# runs under nohup on the pod, so it keeps running if the laptop disconnects.
set -euo pipefail

: "${PRIME_SSH_TARGET:?Set PRIME_SSH_TARGET, for example ubuntu@1.2.3.4.}"
: "${LAGUNA_VLM_ROOT:?Set LAGUNA_VLM_ROOT to the mounted Prime persistent disk path on the pod.}"

SSH_KEY="${SSH_KEY:-${HOME}/.ssh/prime_intellect}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-~/laguna-vision}"
MAX_RUNTIME="${MAX_RUNTIME:-9h}"
RUN_NAME="${RUN_NAME:-laguna-vlm-$(date -u +%Y%m%dT%H%M%SZ)}"
SSH_OPTS=(-i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new)

rsync -az \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude .git \
  --exclude .smoke \
  --exclude data \
  --exclude checkpoints \
  --exclude outputs \
  --exclude runs \
  --exclude venv \
  ./ "${PRIME_SSH_TARGET}:${REMOTE_REPO_DIR}/"

ssh "${SSH_OPTS[@]}" "${PRIME_SSH_TARGET}" "bash -lc '
  set -euo pipefail
  cd ${REMOTE_REPO_DIR}
  if [[ ! -x venv/bin/python ]]; then
    python3 -m venv venv
  fi
  source venv/bin/activate
  python -m pip install -U pip >/tmp/laguna_vision_pip_upgrade.log
  python -m pip install -e \".[llama,data,publish]\" >/tmp/laguna_vision_install.log
  mkdir -p \"${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}\"
  export LAGUNA_VLM_ROOT=\"${LAGUNA_VLM_ROOT}\"
  export MAX_RUNTIME=\"${MAX_RUNTIME}\"
  export RUN_NAME=\"${RUN_NAME}\"
  export STAGE1_MAX_ITEMS=\"${STAGE1_MAX_ITEMS:-300000}\"
  export STAGE2_MAX_ITEMS=\"${STAGE2_MAX_ITEMS:-150000}\"
  export PUBLISH_ON_EXIT=\"${PUBLISH_ON_EXIT:-0}\"
  export HF_REPO_ID=\"${HF_REPO_ID:-}\"
  export HF_PRIVATE=\"${HF_PRIVATE:-1}\"
  export TERMINATE_ON_EXIT=\"${TERMINATE_ON_EXIT:-0}\"
  export PRIME_POD_ID=\"${PRIME_POD_ID:-}\"
  nohup bash scripts/prime_budget_training_job.sh > \"${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}/launcher.log\" 2>&1 < /dev/null &
  echo \$! > \"${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}/pid\"
  echo \"run_name=${RUN_NAME}\"
  echo \"pid=\$(cat \"${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}/pid\")\"
  echo \"log=${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}/job.log\"
'"
