from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EncodedPromptEmbeds:
    data: str
    token_count: int
    embedding_dim: int
    dtype: str


@dataclass(frozen=True)
class GenerationInputs:
    image: Any
    question: str
    context: str = ""
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: Any = None
    request_id: str | None = None


@dataclass(frozen=True)
class LagunaVisionVllmConfig:
    checkpoint: Path | str
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    model: str = ""
    api_key: str | None = None
    vision_device: str = "auto"
    embedding_dtype: str = "float32"
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float | None = None
    timeout: float = 120.0
    allow_local_files: bool = False
