from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

from lagunavision.serve.vllm.service import LagunaVisionVllmService
from lagunavision.serve.vllm.types import EncodedPromptEmbeds, LagunaVisionVllmConfig


class FakeEmbedder:
    projector_path = Path("/tmp/checkpoint/projector.pt")

    async def encode_full_prompt(
        self, image_path: Path, question: str, context: str = "", text_embedder: Any = None,
    ) -> EncodedPromptEmbeds:
        assert image_path.exists()
        return EncodedPromptEmbeds(data="encoded-embeds", token_count=34, embedding_dim=2048, dtype="float32")

    async def encode_image(self, image_path: Path) -> EncodedPromptEmbeds:
        assert image_path.exists()
        return EncodedPromptEmbeds(data="encoded-embeds", token_count=2, embedding_dim=3, dtype="float32")


class FakeTextEmbedder:
    pass


class FakeClient:
    model = "laguna-vision"

    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    async def completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return {"choices": [{"text": "answer text", "index": 0, "finish_reason": "stop"}]}

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return {"choices": [{"message": {"content": "answer text"}}]}


def test_service_builds_completions_request_and_returns_answer() -> None:
    client = FakeClient()
    service = LagunaVisionVllmService(
        config=LagunaVisionVllmConfig(checkpoint="checkpoint", model="laguna-vision"),
        embedder=FakeEmbedder(),
        text_embedder=FakeTextEmbedder(),
        client=client,
    )
    image = base64.b64encode(b"not-a-real-image").decode("ascii")

    response = asyncio.run(service.answer_payload({"inputs": {"image": image, "question": "What is shown?"}}))

    assert response["answer"] == "answer text"
    assert response["backend"] == "vllm"
    assert response["visual_tokens"] == 34
    assert client.payload is not None
    assert "prompt_embeds" in client.payload
    assert "messages" not in client.payload
    assert client.payload["prompt_embeds"] == "encoded-embeds"
