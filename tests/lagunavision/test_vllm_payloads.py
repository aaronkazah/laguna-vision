from __future__ import annotations

import base64

import pytest

from lagunavision.serve.vllm.payloads import (
    build_vllm_completions_payload,
    load_image,
    normalize_generation_inputs,
    request_has_image,
)
from lagunavision.serve.vllm.types import EncodedPromptEmbeds


def test_normalize_hf_style_payload() -> None:
    inputs = normalize_generation_inputs(
        {
            "inputs": {
                "image": "data:image/png;base64,aW1hZ2U=",
                "question": "What is shown?",
                "context": "OCR text",
                "max_new_tokens": 17,
                "temperature": 0.2,
            }
        }
    )

    assert inputs.question == "What is shown?"
    assert inputs.context == "OCR text"
    assert inputs.max_new_tokens == 17
    assert inputs.temperature == 0.2


def test_normalize_openai_style_payload() -> None:
    inputs = normalize_generation_inputs(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Read this screen."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                    ],
                }
            ],
            "max_tokens": 9,
        }
    )

    assert inputs.question == "Read this screen."
    assert inputs.image == "data:image/png;base64,aW1hZ2U="
    assert inputs.max_new_tokens == 9


def test_request_has_image_rejects_text_only_chat() -> None:
    assert not request_has_image({"messages": [{"role": "user", "content": "hello"}]})


def test_load_image_decodes_base64_and_requires_opt_in_for_local_paths(tmp_path) -> None:
    encoded = base64.b64encode(b"image-bytes").decode("ascii")
    loaded = load_image(encoded)
    try:
        assert loaded.cleanup
        assert loaded.path.read_bytes() == b"image-bytes"
    finally:
        loaded.path.unlink(missing_ok=True)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"local-image")
    with pytest.raises(ValueError, match="Local image paths are disabled"):
        load_image(str(image_path))

    local = load_image(str(image_path), allow_local_files=True)
    assert not local.cleanup
    assert local.path == image_path


def test_build_vllm_completions_payload_uses_top_level_prompt_embeds() -> None:
    embeds = EncodedPromptEmbeds(data="encoded-tensor", token_count=34, embedding_dim=2048, dtype="float32")
    payload = build_vllm_completions_payload(
        model="laguna-vision",
        prompt_embeds=embeds,
        max_new_tokens=64,
        temperature=0.0,
    )

    assert payload["model"] == "laguna-vision"
    assert payload["max_tokens"] == 64
    assert payload["prompt_embeds"] == "encoded-tensor"
    assert "messages" not in payload
    assert "prompt" not in payload
