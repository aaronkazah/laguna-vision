from __future__ import annotations

import json

from lagunavision.serve.vllm.embedder import load_checkpoint_projector_spec


def test_load_checkpoint_projector_spec_reads_sidecar_json(tmp_path) -> None:
    checkpoint = tmp_path / "projector.pt"
    checkpoint.write_bytes(b"state")
    (tmp_path / "projector_spec.json").write_text(
        json.dumps(
            {
                "input_dim": 1159,
                "embedding_dim": 4096,
                "hidden_dim": 1024,
                "projector": "resampler",
                "visual_tokens": 256,
                "encoder": "hf",
                "encoder_id": "google/siglip-so400m-patch14-384",
                "max_tiles": 4,
                "patch_px": 14,
                "model_id": "poolside/Laguna-XS.2",
            }
        ),
        encoding="utf-8",
    )

    projector_path, spec, metadata = load_checkpoint_projector_spec(tmp_path)

    assert projector_path == checkpoint
    assert spec.projector == "resampler"
    assert spec.visual_tokens == 256
    assert spec.encoder_id == "google/siglip-so400m-patch14-384"
    assert metadata["model_id"] == "poolside/Laguna-XS.2"
