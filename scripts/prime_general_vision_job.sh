#!/usr/bin/env bash
# Prime-only staged general-vision run: data prep -> feature cache -> alignment -> instruction -> controls.
set -euo pipefail

: "${LAGUNA_VLM_ROOT:?Set LAGUNA_VLM_ROOT to the mounted Prime persistent disk path.}"

if [[ "${ALLOW_POD_LOCAL_OUTPUT:-0}" != "1" && "${LAGUNA_VLM_ROOT}" != /mnt/* && "${LAGUNA_VLM_ROOT}" != /workspace/* && "${LAGUNA_VLM_ROOT}" != /prime/* && "${LAGUNA_VLM_ROOT}" != /data/* ]]; then
  echo "Refusing to write production checkpoints outside a mounted persistent path: ${LAGUNA_VLM_ROOT}" >&2
  echo "Set ALLOW_POD_LOCAL_OUTPUT=1 only for local smoke tests." >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-laguna-general-vision-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_RUNTIME="${MAX_RUNTIME:-36h}"
MODEL_ID="${MODEL_ID:-poolside/Laguna-XS.2}"
VISION_TOWER="${VISION_TOWER:-google/siglip-so400m-patch14-384}"
MAX_TILES="${MAX_TILES:-4}"
VISUAL_TOKENS="${VISUAL_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NPROC="${NPROC:-8}"
DATASET_RECIPE="${DATASET_RECIPE:-general-vision-300k-v1}"
DATASET_SAMPLE_PER_SOURCE="${DATASET_SAMPLE_PER_SOURCE:-0}"
DATASET_TRAIN_BUDGET="${DATASET_TRAIN_BUDGET:-0}"
DATA_DIR="${DATA_DIR:-${LAGUNA_VLM_ROOT}/datasets/${DATASET_RECIPE}}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-${LAGUNA_VLM_ROOT}/feature_cache/${DATASET_RECIPE}-siglip-so400m-tiles${MAX_TILES}}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}/stage1_alignment}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}/stage2_instruction}"
LOG_DIR="${LOG_DIR:-${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}}"
CLI="${LAGUNA_VISION_CLI:-laguna-vision}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-1}"
COCO_TRAIN2017_ROOT="${COCO_TRAIN2017_ROOT:-${LAGUNA_VLM_ROOT}/datasets/raw/coco/train2017}"
LLAVA_PRETRAIN_IMAGE_ROOT="${LLAVA_PRETRAIN_IMAGE_ROOT:-${LAGUNA_VLM_ROOT}/datasets/raw/llava_pretrain}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE1_LR="${STAGE1_LR:-1e-3}"
STAGE1_GRAD_ACCUM="${STAGE1_GRAD_ACCUM:-16}"
STAGE1_SAVE_EVERY="${STAGE1_SAVE_EVERY:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
STAGE2_LR="${STAGE2_LR:-2e-5}"
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-16}"
STAGE2_SAVE_EVERY="${STAGE2_SAVE_EVERY:-100}"
EVAL_LIMIT="${EVAL_LIMIT:-128}"
PUBLISH_ON_EXIT="${PUBLISH_ON_EXIT:-1}"
PUBLISH_DURING_RUN="${PUBLISH_DURING_RUN:-1}"
HF_PUBLISH_INTERVAL="${HF_PUBLISH_INTERVAL:-300}"
HF_UPDATE_LATEST="${HF_UPDATE_LATEST:-1}"
HF_LATEST_PATH_IN_REPO="${HF_LATEST_PATH_IN_REPO:-latest}"
TERMINATE_AFTER_STAGE2_FINAL="${TERMINATE_AFTER_STAGE2_FINAL:-0}"

mkdir -p "${LOG_DIR}" "${STAGE1_OUTPUT_DIR}" "${STAGE2_OUTPUT_DIR}" "${FEATURE_CACHE_ROOT}"
exec > >(tee -a "${LOG_DIR}/job.log") 2>&1

publish_checkpoint() {
  local checkpoint_dir="$1"
  local artifact_name="$2"
  [[ -n "${HF_REPO_ID:-}" ]] || return 0
  [[ -f "${checkpoint_dir}/projector.pt" ]] || return 0
  [[ -f "${checkpoint_dir}/projector_spec.json" ]] || return 0
  [[ -f "${checkpoint_dir}/train_report.json" ]] || return 0

  CHECKPOINT_DIR="${checkpoint_dir}" \
    HF_PRIVATE="${HF_PRIVATE:-1}" \
    PATH_IN_REPO="${HF_PATH_IN_REPO:-${RUN_NAME}}/${artifact_name}" \
    scripts/publish_hf_checkpoint.sh

  if [[ "${HF_UPDATE_LATEST}" == "1" && "${artifact_name}" == "stage2_final" ]]; then
    CHECKPOINT_DIR="${checkpoint_dir}" \
      HF_PRIVATE="${HF_PRIVATE:-1}" \
      HF_REPLACE_PATH=1 \
      PATH_IN_REPO="${HF_LATEST_PATH_IN_REPO}" \
      scripts/publish_hf_checkpoint.sh
  fi
}

publish_new_checkpoints() {
  local watched_pid="$1"
  local published_dir="${LOG_DIR}/published_hf"
  mkdir -p "${published_dir}"
  while kill -0 "${watched_pid}" >/dev/null 2>&1; do
    for checkpoint_dir in "${STAGE1_OUTPUT_DIR}"/step_* "${STAGE2_OUTPUT_DIR}"/step_*; do
      [[ -d "${checkpoint_dir}" ]] || continue
      [[ -f "${checkpoint_dir}/projector.pt" ]] || continue
      local stage
      if [[ "${checkpoint_dir}" == "${STAGE1_OUTPUT_DIR}"/* ]]; then
        stage="stage1"
      else
        stage="stage2"
      fi
      local marker="${published_dir}/${stage}_$(basename "${checkpoint_dir}")"
      [[ ! -f "${marker}" ]] || continue
      publish_checkpoint "${checkpoint_dir}" "${stage}/$(basename "${checkpoint_dir}")" && touch "${marker}"
    done
    sleep "${HF_PUBLISH_INTERVAL}"
  done
  for checkpoint_dir in "${STAGE1_OUTPUT_DIR}"/step_* "${STAGE2_OUTPUT_DIR}"/step_*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    [[ -f "${checkpoint_dir}/projector.pt" ]] || continue
    local stage
    if [[ "${checkpoint_dir}" == "${STAGE1_OUTPUT_DIR}"/* ]]; then
      stage="stage1"
    else
      stage="stage2"
    fi
    local marker="${published_dir}/${stage}_$(basename "${checkpoint_dir}")"
    [[ ! -f "${marker}" ]] || continue
    publish_checkpoint "${checkpoint_dir}" "${stage}/$(basename "${checkpoint_dir}")" && touch "${marker}"
  done
}

publish_run_metadata() {
  [[ -n "${HF_REPO_ID:-}" ]] || return 0
  python3 - <<'PY' || true
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
private = os.environ.get("HF_PRIVATE", "1").lower() not in {"0", "false", "no", "public"}
base = os.environ.get("HF_PATH_IN_REPO") or os.environ["RUN_NAME"]
run_name = os.environ["RUN_NAME"]
data_dir = Path(os.environ["DATA_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
train_budget = int(os.environ.get("DATASET_TRAIN_BUDGET") or 0)
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
state_path = log_dir / "run_state.json"
state_path.write_text(
    json.dumps(
        {
            "run_name": run_name,
            "dataset_recipe": os.environ["DATASET_RECIPE"],
            "data_dir": os.environ["DATA_DIR"],
            "feature_cache_root": os.environ["FEATURE_CACHE_ROOT"],
            "stage1_output_dir": os.environ["STAGE1_OUTPUT_DIR"],
            "stage2_output_dir": os.environ["STAGE2_OUTPUT_DIR"],
            "max_tiles": os.environ["MAX_TILES"],
            "note": "Raw datasets are intentionally not uploaded; recreate data with the recipe and public sources.",
        },
        indent=2,
    ),
    encoding="utf-8",
)
model_card_path = log_dir / "README.md"
model_card_path.write_text(
    f"""---
license: other
base_model: poolside/Laguna-XS.2
library_name: transformers
tags:
- vision-language
- laguna
- siglip
- llava
- lora
pipeline_tag: image-to-text
private: {str(private).lower()}
---

# Laguna Vision adapter

This repository contains Laguna Vision checkpoint artifacts for `{run_name}`. The model is an early visual adapter for `poolside/Laguna-XS.2`, using `google/siglip-so400m-patch14-384` image features, AnyRes tiling, a resampler projector, and optional LoRA instruction adapters.

The project was built for the [Poolside Research Hackathon](https://www.competehub.dev/en/competitions/lumaevt-toewzCfp1Ue1PcR) as a near-capability exploration for computer use: visual grounding for screenshots, UI state, code/debug images, documents, and other context that a text-only Laguna prompt cannot inspect directly.

## Training recipe

- Recipe: `{os.environ["DATASET_RECIPE"]}`
- Train budget: `{train_budget or "full recipe"}` examples
- Max image tiles: `{os.environ["MAX_TILES"]}` global-plus-crop tiles
- Stage 1: visual projector alignment, projector only
- Stage 2: projector plus LoRA instruction tuning
- Checkpoints: saved every 100 optimizer steps under `{base}/stage1/...` and `{base}/stage2/...`

The 200k hackathon release used 80k alignment examples from LLaVA pretrain and 120k instruction examples from LLaVA Instruct, ShareGPT4V, GQA, DocumentVQA, TextVQA, OCR-VQA, ChartQA, WebSight, RICO ScreenQA, RICO Screen2Words, WebSRC, and synthetic spatial OCR. The full locked recipe remains 300k examples; the 200k slice was a time/budget concession to prove the complete training, serving, and evaluation path.

## Data handling

Raw public datasets and image archives are intentionally not uploaded to this Hugging Face model repo. The Prime run keeps raw assets, manifests, feature caches, logs, and checkpoints on the persistent `/data/laguna-vlm` disk. This repo uploads model artifacts plus reproducibility metadata only.

Uploaded metadata for this run lives at `{base}/run_metadata/`.

## API input format

The included `handler.py` accepts either a simple JSON payload:

```json
{{"inputs": {{"image": "data:image/png;base64,...", "question": "What is shown?"}}}}
```

or OpenAI-style multimodal messages:

```json
{{"inputs": {{"messages": [{{"role": "user", "content": [{{"type": "text", "text": "What is shown?"}}, {{"type": "image_url", "image_url": {{"url": "data:image/png;base64,..."}}}}]}}]}}}}
```

`image`/`image_url.url` may be a base64 string, data URI, HTTPS URL, or local file path available to the endpoint.

## Intended use

This is an experimental checkpoint intended to verify that Laguna can condition on images and answer broadly across natural images, screenshots, OCR, documents, charts, UI, and spatial questions. It is not a safety-reviewed production model.

## Current evaluation status

Serving works, but the latest checkpoint is only weakly grounded. In the live capability probe it passed 1 / 7 cases: no-visible-text abstention. It failed simple color/shape controls, precise OCR, dense UI state, table values, meme semantics, counting, and exact localization. Use `laguna-vision capability-probe --output-dir data/capability_probe` to generate deterministic expected-pass and known-failure probes, then run `laguna-vision eval-endpoint --endpoint <url> --manifest data/capability_probe/manifest.jsonl --output runs/evals/capability_probe.answers.jsonl` to score a deployed endpoint by category.

## Limitations

- Early checkpoint quality is uneven.
- OCR, dense UI localization, meme semantics, counting, and precise spatial localization can fail.
- The adapter depends on the Laguna Vision inference/training code in this repository layout.
- Generated text may hallucinate; validate outputs before relying on them.
""",
    encoding="utf-8",
)
for local, name in ((data_dir / "recipe.json", "recipe.json"), (state_path, "run_state.json"), (log_dir / "job.log", "job.log")):
    if local.exists():
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(local),
            path_in_repo=f"{base}/run_metadata/{name}",
        )
api.upload_file(
    repo_id=repo_id,
    repo_type="model",
    path_or_fileobj=str(model_card_path),
    path_in_repo="README.md",
)
api.upload_file(
    repo_id=repo_id,
    repo_type="model",
    path_or_fileobj=str(model_card_path),
    path_in_repo=f"{base}/run_metadata/MODEL_CARD.md",
)
for local, remote in (
    (Path("huggingface/handler.py"), "handler.py"),
    (Path("huggingface/requirements.txt"), "requirements.txt"),
):
    if local.exists():
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(local),
            path_in_repo=remote,
        )
PY
}

finish_on_exit() {
  local status=$?
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_status=${status}"
  if [[ "${PUBLISH_ON_EXIT}" == "1" ]]; then
    publish_checkpoint "${STAGE1_OUTPUT_DIR}" stage1_final || true
    publish_checkpoint "${STAGE2_OUTPUT_DIR}" stage2_final || true
    publish_run_metadata
  fi
  if [[ "${TERMINATE_ON_EXIT:-0}" == "1" || "${TERMINATE_AFTER_STAGE2_FINAL}" == "1" ]]; then
    echo "generic_auto_termination_removed=1; pod termination is allowed only after explicit stage2 final checkpoint verification."
  fi
  exit "${status}"
}
trap finish_on_exit EXIT

terminate_after_stage2_final() {
  [[ "${TERMINATE_AFTER_STAGE2_FINAL}" == "1" ]] || return 0
  if [[ ! -f "${STAGE2_OUTPUT_DIR}/projector.pt" || ! -f "${STAGE2_OUTPUT_DIR}/projector_spec.json" || ! -f "${STAGE2_OUTPUT_DIR}/train_report.json" ]]; then
    echo "stage2_final_incomplete; leaving pod running."
    return 0
  fi
  publish_checkpoint "${STAGE2_OUTPUT_DIR}" stage2_final || true
  publish_run_metadata
  if [[ -z "${PRIME_POD_ID:-}" ]]; then
    echo "TERMINATE_AFTER_STAGE2_FINAL=1 but PRIME_POD_ID is unset; leaving pod running."
  elif command -v prime >/dev/null 2>&1; then
    echo "terminating_after_verified_stage2_final_checkpoint=${STAGE2_OUTPUT_DIR}"
    prime pods terminate "${PRIME_POD_ID}" -y || true
  else
    echo "TERMINATE_AFTER_STAGE2_FINAL=1 but prime CLI is not installed in the pod; leaving pod running."
  fi
}

echo "run_name=${RUN_NAME}"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "max_runtime=${MAX_RUNTIME}"
echo "dataset_recipe=${DATASET_RECIPE}"
echo "data_dir=${DATA_DIR}"
echo "feature_cache_root=${FEATURE_CACHE_ROOT}"
echo "stage1_output_dir=${STAGE1_OUTPUT_DIR}"
echo "stage2_output_dir=${STAGE2_OUTPUT_DIR}"
echo "model_id=${MODEL_ID}"
echo "vision_tower=${VISION_TOWER}"
echo "max_tiles=${MAX_TILES}"
echo "visual_tokens=${VISUAL_TOKENS}"
echo "nproc=${NPROC}"
echo "publish_during_run=${PUBLISH_DURING_RUN}"
echo "hf_publish_interval=${HF_PUBLISH_INTERVAL}"

body_script="${LOG_DIR}/run_general_vision_body.sh"
cat > "${body_script}" <<'GENERAL_VISION_BODY'
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

manifests_ready() {
  [[ -f "${DATA_DIR}/recipe.json" ]] || return 1
  python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
with (data_dir / "recipe.json").open(encoding="utf-8") as handle:
    recipe = json.load(handle)
required = {
    "alignment/train.jsonl": recipe["alignment_train"],
    "alignment/eval.jsonl": recipe["alignment_eval"],
    "instruction/train.jsonl": recipe["instruction_train"],
    "instruction/eval.jsonl": recipe["instruction_eval"],
    "controls/wrong.jsonl": recipe["instruction_eval"],
    "controls/blank.jsonl": recipe["instruction_eval"],
}
for relative, expected in required.items():
    path = data_dir / relative
    if not path.exists():
        print(f"manifest_missing path={path}", file=sys.stderr)
        sys.exit(1)
    with path.open(encoding="utf-8") as handle:
        actual = sum(1 for _ in handle)
    if actual != expected:
        print(f"manifest_incomplete path={path} actual={actual} expected={expected}", file=sys.stderr)
        sys.exit(1)
PY
}

materialize_args=(
  general-materialize
  --output-dir "${DATA_DIR}"
  --recipe "${DATASET_RECIPE}"
  --coco-train2017-root "${COCO_TRAIN2017_ROOT}"
  --llava-pretrain-image-root "${LLAVA_PRETRAIN_IMAGE_ROOT}"
)
if [[ "${DOWNLOAD_ASSETS}" == "1" ]]; then
  materialize_args+=(--download-assets)
fi
if (( DATASET_SAMPLE_PER_SOURCE > 0 )); then
  materialize_args+=(--sample-per-source "${DATASET_SAMPLE_PER_SOURCE}")
fi
if (( DATASET_TRAIN_BUDGET > 0 )); then
  materialize_args+=(--train-budget "${DATASET_TRAIN_BUDGET}")
fi

if manifests_ready; then
  echo "using_cached_general_manifests data_dir=${DATA_DIR}"
else
  if ! "${CLI}" "${materialize_args[@]}"; then
    if manifests_ready; then
      echo "materialize_returned_nonzero_but_manifests_ready=1"
    else
      exit 1
    fi
  fi
fi

cache_manifest() {
  local manifest="$1"
  local feature_dir="$2"
  export MANIFEST="${manifest}"
  export FEATURE_CACHE_DIR="${feature_dir}"
  export VISION_TOWER="${VISION_TOWER}"
  export MAX_TILES="${MAX_TILES}"
  export NPROC="${NPROC}"
  scripts/laguna_llava_cache_features.sh
}

cache_manifest "${DATA_DIR}/alignment/train.jsonl" "${FEATURE_CACHE_ROOT}/alignment"
cache_manifest "${DATA_DIR}/alignment/eval.jsonl" "${FEATURE_CACHE_ROOT}/alignment"
cache_manifest "${DATA_DIR}/instruction/train.jsonl" "${FEATURE_CACHE_ROOT}/instruction"
cache_manifest "${DATA_DIR}/instruction/eval.jsonl" "${FEATURE_CACHE_ROOT}/instruction"

latest_step_dir() {
  local output_dir="$1"
  find "${output_dir}" -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort | tail -1
}

if [[ -f "${STAGE1_OUTPUT_DIR}/projector.pt" ]]; then
  echo "stage1_already_complete checkpoint=${STAGE1_OUTPUT_DIR}/projector.pt"
else
  stage1_resume="$(latest_step_dir "${STAGE1_OUTPUT_DIR}")"
  unset INIT_CHECKPOINT
  if [[ -n "${stage1_resume}" && -f "${stage1_resume}/projector.pt" ]]; then
    export INIT_CHECKPOINT="${stage1_resume}/projector.pt"
    echo "stage1_resuming_from=${stage1_resume}/projector.pt"
  fi
  MANIFEST="${DATA_DIR}/alignment/train.jsonl" \
  EVAL_MANIFEST="${DATA_DIR}/alignment/eval.jsonl" \
  OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" \
  FEATURE_CACHE_DIR="${FEATURE_CACHE_ROOT}/alignment" \
  MODEL_ID="${MODEL_ID}" \
  VISION_TOWER="${VISION_TOWER}" \
  MAX_TILES="${MAX_TILES}" \
  VISUAL_TOKENS="${VISUAL_TOKENS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM="${STAGE1_GRAD_ACCUM}" \
  EPOCHS="${STAGE1_EPOCHS}" \
  LR="${STAGE1_LR}" \
  SAVE_EVERY="${STAGE1_SAVE_EVERY}" \
  NPROC="${NPROC}" \
  scripts/laguna_llava_stage1.sh
fi

if [[ -f "${STAGE2_OUTPUT_DIR}/projector.pt" ]]; then
  echo "stage2_already_complete checkpoint=${STAGE2_OUTPUT_DIR}/projector.pt"
else
  stage2_resume="$(latest_step_dir "${STAGE2_OUTPUT_DIR}")"
  unset INIT_CHECKPOINT INIT_LORA_DIR
  export STAGE1_CHECKPOINT="${STAGE1_OUTPUT_DIR}/projector.pt"
  if [[ -n "${stage2_resume}" && -f "${stage2_resume}/projector.pt" ]]; then
    unset STAGE1_CHECKPOINT
    export INIT_CHECKPOINT="${stage2_resume}/projector.pt"
    if [[ -d "${stage2_resume}/lora" ]]; then
      export INIT_LORA_DIR="${stage2_resume}/lora"
    fi
    echo "stage2_resuming_from=${stage2_resume}"
  fi
  MANIFEST="${DATA_DIR}/instruction/train.jsonl" \
  EVAL_MANIFEST="${DATA_DIR}/instruction/eval.jsonl" \
  OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" \
  FEATURE_CACHE_DIR="${FEATURE_CACHE_ROOT}/instruction" \
  MODEL_ID="${MODEL_ID}" \
  VISION_TOWER="${VISION_TOWER}" \
  MAX_TILES="${MAX_TILES}" \
  VISUAL_TOKENS="${VISUAL_TOKENS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM="${STAGE2_GRAD_ACCUM}" \
  EPOCHS="${STAGE2_EPOCHS}" \
  LR="${STAGE2_LR}" \
  SAVE_EVERY="${STAGE2_SAVE_EVERY}" \
  NPROC="${NPROC}" \
  scripts/laguna_llava_stage2.sh
fi

run_ablation() {
  local manifest="$1"
  local output="$2"
  eval_args=(
    eval-ablation
    --manifest "${manifest}"
    --checkpoint "${STAGE2_OUTPUT_DIR}/projector.pt"
    --output "${output}"
    --backbone laguna
    --model-id "${MODEL_ID}"
    --threshold 0.15
    --device cuda
    --vision-device cuda
  )
  if (( EVAL_LIMIT > 0 )); then
    eval_args+=(--limit "${EVAL_LIMIT}")
  fi
  "${CLI}" "${eval_args[@]}"
}

if [[ -f "${STAGE2_OUTPUT_DIR}/projector.pt" ]]; then
  run_ablation "${DATA_DIR}/instruction/eval.jsonl" "${STAGE2_OUTPUT_DIR}/ablation_correct.jsonl"
  run_ablation "${DATA_DIR}/controls/wrong.jsonl" "${STAGE2_OUTPUT_DIR}/ablation_wrong.jsonl"
  run_ablation "${DATA_DIR}/controls/blank.jsonl" "${STAGE2_OUTPUT_DIR}/ablation_blank.jsonl"
fi

stage1_train_rows="$(line_count "${DATA_DIR}/alignment/train.jsonl")"
stage1_eval_rows="$(line_count "${DATA_DIR}/alignment/eval.jsonl")"
stage2_train_rows="$(line_count "${DATA_DIR}/instruction/train.jsonl")"
stage2_eval_rows="$(line_count "${DATA_DIR}/instruction/eval.jsonl")"
echo "stage1_rows train=${stage1_train_rows} eval=${stage1_eval_rows}"
echo "stage2_rows train=${stage2_train_rows} eval=${stage2_eval_rows}"
terminate_after_stage2_final
GENERAL_VISION_BODY
chmod +x "${body_script}"

export BATCH_SIZE CLI COCO_TRAIN2017_ROOT DATA_DIR DATASET_RECIPE DATASET_SAMPLE_PER_SOURCE DATASET_TRAIN_BUDGET DOWNLOAD_ASSETS EVAL_LIMIT
export FEATURE_CACHE_ROOT LLAVA_PRETRAIN_IMAGE_ROOT MAX_TILES MODEL_ID NPROC STAGE1_EPOCHS STAGE1_GRAD_ACCUM STAGE1_LR
export STAGE1_OUTPUT_DIR STAGE1_SAVE_EVERY STAGE2_EPOCHS STAGE2_GRAD_ACCUM STAGE2_LR STAGE2_OUTPUT_DIR STAGE2_SAVE_EVERY
export VISION_TOWER VISUAL_TOKENS
export RUN_NAME LOG_DIR HF_REPO_ID HF_PRIVATE HF_PATH_IN_REPO HF_LATEST_PATH_IN_REPO HF_UPDATE_LATEST

timeout "${MAX_RUNTIME}" bash "${body_script}" &
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
