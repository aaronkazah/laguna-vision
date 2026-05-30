from __future__ import annotations

import os
import warnings
from typing import Any

from lagunavision.serve.vllm.client import VllmOpenAIClient
from lagunavision.serve.vllm.embedder import LagunaVisionPromptEmbedder, TextEmbedder, load_checkpoint_projector_spec
from lagunavision.serve.vllm.payloads import (
    build_vllm_completions_payload,
    extract_openai_answer,
    load_image,
    normalize_generation_inputs,
)
from lagunavision.serve.vllm.types import GenerationInputs, LagunaVisionVllmConfig


class LagunaVisionVllmService:
    def __init__(
        self,
        *,
        config: LagunaVisionVllmConfig,
        embedder: LagunaVisionPromptEmbedder,
        text_embedder: TextEmbedder,
        client: VllmOpenAIClient,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.text_embedder = text_embedder
        self.client = client
        self.checkpoint = str(embedder.projector_path.parent)

    @classmethod
    def from_config(cls, config: LagunaVisionVllmConfig) -> "LagunaVisionVllmService":
        projector_path, spec, checkpoint_row = load_checkpoint_projector_spec(config.checkpoint)
        model_id = (
            config.model
            or os.environ.get("VLLM_MODEL")
            or checkpoint_row.get("vllm_model")
            or checkpoint_row.get("model_id")
            or ""
        )
        embedder = LagunaVisionPromptEmbedder.from_checkpoint(
            projector_path,
            device=config.vision_device,
            output_dtype=config.embedding_dtype,
        )
        backbone_id = checkpoint_row.get("model_id", model_id)
        text_embedder = TextEmbedder.from_model_id(str(backbone_id), device="cpu")
        client = VllmOpenAIClient(
            base_url=config.vllm_base_url,
            model=str(model_id),
            api_key=config.api_key,
            timeout=config.timeout,
        )
        return cls(config=config, embedder=embedder, text_embedder=text_embedder, client=client)

    async def answer_inputs(self, inputs: GenerationInputs, *, return_raw: bool = False) -> dict[str, Any]:
        loaded = load_image(inputs.image, allow_local_files=self.config.allow_local_files)
        try:
            prompt_embeds = await self.embedder.encode_full_prompt(
                loaded.path,
                question=inputs.question,
                context=inputs.context,
                text_embedder=self.text_embedder,
            )
        finally:
            if loaded.cleanup:
                try:
                    loaded.path.unlink(missing_ok=True)
                except OSError as exc:
                    warnings.warn(f"Could not remove temporary image {loaded.path}: {exc}", RuntimeWarning)

        payload = build_vllm_completions_payload(
            model=self.client.model,
            prompt_embeds=prompt_embeds,
            max_new_tokens=inputs.max_new_tokens or self.config.max_new_tokens,
            temperature=self.config.temperature if inputs.temperature is None else inputs.temperature,
            top_p=self.config.top_p if inputs.top_p is None else inputs.top_p,
            stop=inputs.stop,
        )
        raw_response = await self.client.completion(payload)
        if return_raw:
            return raw_response

        return {
            "answer": extract_openai_answer(raw_response),
            "checkpoint": self.checkpoint,
            "backend": "vllm",
            "model": self.client.model,
            "visual_tokens": prompt_embeds.token_count,
            "embedding_dim": prompt_embeds.embedding_dim,
            "embedding_dtype": prompt_embeds.dtype,
            "request_id": inputs.request_id,
        }

    async def answer_payload(self, payload: dict[str, Any], *, return_raw: bool = False) -> dict[str, Any]:
        return await self.answer_inputs(normalize_generation_inputs(payload), return_raw=return_raw)
