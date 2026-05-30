"""vLLM serving path for Laguna Vision.

The package keeps the production serving integration separate from the
training code and the Hugging Face endpoint handler. vLLM owns text generation;
Laguna Vision owns image tiling, SigLIP features, and projector embeddings.
"""

from __future__ import annotations

from lagunavision.serve.vllm.client import VllmOpenAIClient
from lagunavision.serve.vllm.payloads import (
    build_vllm_completions_payload,
    extract_openai_answer,
    normalize_generation_inputs,
)
from lagunavision.serve.vllm.service import LagunaVisionVllmService
from lagunavision.serve.vllm.smoke import smoke_prompt_embeds
from lagunavision.serve.vllm.types import (
    EncodedPromptEmbeds,
    GenerationInputs,
    LagunaVisionVllmConfig,
)

__all__ = [
    "EncodedPromptEmbeds",
    "GenerationInputs",
    "LagunaVisionVllmConfig",
    "LagunaVisionVllmService",
    "VllmOpenAIClient",
    "build_vllm_completions_payload",
    "extract_openai_answer",
    "normalize_generation_inputs",
    "smoke_prompt_embeds",
]
