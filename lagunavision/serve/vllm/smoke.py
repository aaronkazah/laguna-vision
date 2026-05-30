from __future__ import annotations

import asyncio
from typing import Any

from lagunavision.serve.vllm.client import VllmOpenAIClient
from lagunavision.serve.vllm.embedder import _patch_transformers_remote_code_compat, tensor_to_vllm_base64


async def smoke_prompt_embeds(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    embedding_dim: int | None = None,
    hf_model: str | None = None,
    token_count: int = 4,
    max_tokens: int = 8,
    timeout: float = 120.0,
) -> dict[str, Any]:
    resolved_dim = embedding_dim or _hidden_size_from_config(hf_model or model)
    if resolved_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if token_count <= 0:
        raise ValueError("token_count must be positive")

    import torch

    generator = torch.Generator(device="cpu").manual_seed(0)
    embeds = torch.randn(token_count, resolved_dim, generator=generator, dtype=torch.float32) * 0.01
    client = VllmOpenAIClient(base_url=base_url, model=model, api_key=api_key, timeout=timeout)
    response = await client.completion(
        {
            "model": model,
            "prompt_embeds": tensor_to_vllm_base64(embeds),
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    )
    answer = _extract_answer(response)
    if not answer:
        raise RuntimeError(f"vLLM returned no content: {response!r}")
    return {
        "status": "ok",
        "model": model,
        "base_url": base_url,
        "embedding_dim": resolved_dim,
        "prompt_embed_tokens": token_count,
        "answer": answer,
    }


def run_smoke_prompt_embeds(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(smoke_prompt_embeds(**kwargs))


def _hidden_size_from_config(model: str) -> int:
    try:
        _patch_transformers_remote_code_compat()
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("Install transformers or pass --embedding-dim explicitly.") from exc

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not infer hidden_size from {model}; pass --embedding-dim.")
    return int(hidden_size)


def _extract_answer(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    text = first.get("text")
    return text.strip() if isinstance(text, str) else ""
