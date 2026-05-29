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

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit("Install publishing support with `python -m pip install -e '.[publish]'`.") from exc

repo_id = os.environ["HF_REPO_ID"]
checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"]).expanduser().resolve()
path_in_repo = os.environ["PATH_IN_REPO"].strip("/")
revision = os.environ["HF_REVISION"]
private = os.environ["HF_PRIVATE"].lower() not in {"0", "false", "no", "public"}

required = ["projector.pt", "projector_spec.json", "train_report.json"]
missing = [name for name in required if not (checkpoint_dir / name).exists()]
if missing:
    raise SystemExit(f"{checkpoint_dir} is missing required checkpoint files: {', '.join(missing)}")

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(checkpoint_dir),
    path_in_repo=path_in_repo,
    revision=revision,
    ignore_patterns=["*.tmp", "__pycache__/*", ".DS_Store"],
)
print({"repo_id": repo_id, "path_in_repo": path_in_repo, "private": private, "revision": revision})
PY
