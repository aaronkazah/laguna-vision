#!/usr/bin/env bash
# Publish a Laguna Vision checkpoint directory to a Hugging Face model repo.
# Use a private repo for early checkpoints unless the model card/license is ready.
set -euo pipefail

: "${HF_REPO_ID:?Set HF_REPO_ID, for example org/laguna-vision-early.}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a directory containing projector.pt and projector_spec.json.}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PATH_IN_REPO="${PATH_IN_REPO:-$(basename "${CHECKPOINT_DIR}")}"
HF_PRIVATE="${HF_PRIVATE:-1}"
HF_REVISION="${HF_REVISION:-main}"
HF_REPLACE_PATH="${HF_REPLACE_PATH:-0}"
export PATH_IN_REPO HF_PRIVATE HF_REVISION HF_REPLACE_PATH

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as exc:
    raise SystemExit("Install publishing support with `python -m pip install -e '.[publish]'`.") from exc

repo_id = os.environ["HF_REPO_ID"]
checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"]).expanduser().resolve()
path_in_repo = os.environ["PATH_IN_REPO"].strip("/")
revision = os.environ["HF_REVISION"]
private = os.environ["HF_PRIVATE"].lower() not in {"0", "false", "no", "public"}
replace_path = os.environ["HF_REPLACE_PATH"].lower() in {"1", "true", "yes"}

required = ["projector.pt", "projector_spec.json", "train_report.json"]
missing = [name for name in required if not (checkpoint_dir / name).exists()]
if missing:
    raise SystemExit(f"{checkpoint_dir} is missing required checkpoint files: {', '.join(missing)}")

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
if replace_path:
    try:
        api.delete_folder(repo_id=repo_id, repo_type="model", path_in_repo=path_in_repo, revision=revision)
    except HfHubHTTPError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(checkpoint_dir),
    path_in_repo=path_in_repo,
    revision=revision,
    allow_patterns=[
        "projector.pt",
        "projector_spec.json",
        "train_report.json",
        "lora/adapter_config.json",
        "lora/adapter_model.safetensors",
    ],
)
print({"repo_id": repo_id, "path_in_repo": path_in_repo, "private": private, "revision": revision})
PY
