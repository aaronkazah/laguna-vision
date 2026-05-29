from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class HubCheckpointRef:
    repo_id: str
    checkpoint_path: str
    revision: str = "main"

    @property
    def checkpoint_dir(self) -> str:
        parent = str(Path(self.checkpoint_path).parent)
        return "" if parent == "." else parent


Downloader = Callable[..., str]


def resolve_checkpoint_reference(reference: str | Path, *, downloader: Downloader | None = None) -> Path:
    """Resolve a local checkpoint path or an HF Hub checkpoint reference."""
    value = str(reference)
    hub_ref = parse_hub_checkpoint_ref(value)
    if hub_ref is None:
        path = Path(value).expanduser()
        return path / "projector.pt" if path.is_dir() else path

    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install Hub support with `python -m pip install -e '.[publish]'`.") from exc

        downloader = snapshot_download

    prefix = hub_ref.checkpoint_dir
    allow_patterns = _checkpoint_allow_patterns(prefix)
    snapshot_dir = Path(
        downloader(
            repo_id=hub_ref.repo_id,
            repo_type="model",
            revision=hub_ref.revision,
            allow_patterns=allow_patterns,
        )
    )
    checkpoint = snapshot_dir / hub_ref.checkpoint_path
    if not checkpoint.exists():
        raise FileNotFoundError(f"{value} did not resolve to {checkpoint}")
    spec = checkpoint.parent / "projector_spec.json"
    if not spec.exists():
        raise FileNotFoundError(f"{value} is missing required projector_spec.json beside projector.pt")
    return checkpoint


def parse_hub_checkpoint_ref(value: str) -> HubCheckpointRef | None:
    if value.startswith("hf://"):
        return _parse_hf_uri(value.removeprefix("hf://"))
    if ":" in value and not Path(value).exists():
        repo, checkpoint = value.split(":", 1)
        if "/" in repo and checkpoint:
            return HubCheckpointRef(repo_id=repo, checkpoint_path=_projector_path(checkpoint))
    return None


def _parse_hf_uri(value: str) -> HubCheckpointRef:
    path_part, revision = _split_revision(value)
    parts = [part for part in path_part.split("/") if part]
    if len(parts) < 2:
        raise ValueError("HF checkpoint references must look like hf://owner/repo[/checkpoint-dir]")
    repo_id = "/".join(parts[:2])
    checkpoint = "/".join(parts[2:]) if len(parts) > 2 else "projector.pt"
    return HubCheckpointRef(repo_id=repo_id, checkpoint_path=_projector_path(checkpoint), revision=revision)


def _split_revision(value: str) -> tuple[str, str]:
    if "@" not in value:
        return value, "main"
    path, revision = value.rsplit("@", 1)
    return path, revision or "main"


def _projector_path(path: str) -> str:
    cleaned = path.strip("/")
    if not cleaned:
        return "projector.pt"
    return cleaned if cleaned.endswith(".pt") else f"{cleaned}/projector.pt"


def _checkpoint_allow_patterns(prefix: str) -> list[str]:
    if not prefix:
        return ["projector.pt", "projector_spec.json", "train_report.json", "lora/*", "*.jsonl", "*.json"]
    return [
        f"{prefix}/projector.pt",
        f"{prefix}/projector_spec.json",
        f"{prefix}/train_report.json",
        f"{prefix}/lora/*",
        f"{prefix}/*.jsonl",
        f"{prefix}/*.json",
    ]
