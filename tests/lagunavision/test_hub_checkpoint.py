from pathlib import Path

from lagunavision.hub import parse_hub_checkpoint_ref, resolve_checkpoint_reference


def test_resolve_local_checkpoint_directory(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "step_000250"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "projector.pt"
    checkpoint.write_bytes(b"")

    assert resolve_checkpoint_reference(checkpoint_dir) == checkpoint


def test_parse_hf_checkpoint_reference_with_directory() -> None:
    ref = parse_hub_checkpoint_ref("hf://owner/model/checkpoints/step_000250@dev")

    assert ref is not None
    assert ref.repo_id == "owner/model"
    assert ref.checkpoint_path == "checkpoints/step_000250/projector.pt"
    assert ref.revision == "dev"


def test_resolve_hf_checkpoint_reference(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    checkpoint_dir = snapshot / "checkpoints" / "step_000250"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "projector.pt").write_bytes(b"")
    (checkpoint_dir / "projector_spec.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_downloader(**kwargs) -> str:
        calls.append(kwargs)
        return str(snapshot)

    checkpoint = resolve_checkpoint_reference(
        "hf://owner/model/checkpoints/step_000250",
        downloader=fake_downloader,
    )

    assert checkpoint == checkpoint_dir / "projector.pt"
    assert calls == [
        {
            "repo_id": "owner/model",
            "repo_type": "model",
            "revision": "main",
            "allow_patterns": [
                "checkpoints/step_000250/projector.pt",
                "checkpoints/step_000250/projector_spec.json",
                "checkpoints/step_000250/train_report.json",
                "checkpoints/step_000250/lora/*",
                "checkpoints/step_000250/*.jsonl",
                "checkpoints/step_000250/*.json",
            ],
        }
    ]
