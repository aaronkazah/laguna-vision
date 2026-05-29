from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    context: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.0


@dataclass(frozen=True)
class VisualGenerationRequest(GenerationRequest):
    visual_embeddings: Any = None


@dataclass(frozen=True)
class LoraSettings:
    """Low-rank adapter configuration for fine-tuning a frozen backbone.

    ``targets`` empty means every linear layer is adapted; otherwise it names the
    module suffixes to wrap (for example ``("q_proj", "v_proj")``).
    """

    rank: int
    alpha: int = 16
    dropout: float = 0.05
    targets: tuple[str, ...] = field(default_factory=tuple)


class Backbone(ABC):
    """A language-model backbone the visual bridge injects embeddings into.

    Concrete adapters wrap a specific model family but expose one contract, so
    the encoder, projector, trainer, and eval surfaces stay model-agnostic. New
    models are added by registering an adapter, never by branching on model id.
    """

    model_id: str
    device: str

    @abstractmethod
    async def load(self) -> None:
        """Load weights and tokenizer. Idempotent."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Embedding dimension visual tokens must be projected into."""

    @property
    @abstractmethod
    def torch_module(self) -> Any:
        """Underlying trainable nn.Module, exposed so the trainer can backprop."""

    @property
    @abstractmethod
    def resolved_device(self) -> Any:
        """Concrete torch device the loaded weights live on."""

    @abstractmethod
    def freeze(self) -> None:
        """Put the backbone in eval mode and disable gradients on its weights."""

    @abstractmethod
    def enable_lora(self, settings: LoraSettings) -> list[Any]:
        """Attach LoRA adapters to the frozen backbone and return their trainable parameters.

        Called after :meth:`freeze`, so only the returned adapter parameters carry
        gradients; the base weights stay frozen.
        """

    @abstractmethod
    def save_adapter(self, directory: Path) -> None:
        """Persist the attached LoRA adapter weights."""

    @abstractmethod
    def load_adapter(self, directory: Path, *, trainable: bool = False) -> None:
        """Load previously saved LoRA adapter weights for inference or resume training."""

    def adapter_disabled(self) -> AbstractContextManager[None]:
        """Disable any attached adapter for the duration of the context.

        Backbones without an adapter return a null context, so callers can use this
        uniformly to obtain base-weights-only behavior.
        """
        return nullcontext()

    @abstractmethod
    def tokenize_example(self, question: str, answer: str, context: str = "") -> tuple[Any, Any]:
        """Tokenize one example once, returning ``(prompt_ids, answer_ids)`` tensors."""

    @abstractmethod
    def embed_tokens(self, token_ids: Any) -> Any:
        """Map token ids to input embeddings via the backbone's embedding table."""

    @abstractmethod
    def visual_training_batch(
        self, prompt_ids: list[Any], answer_ids: list[Any], visual_embeddings: list[Any]
    ) -> dict[str, Any]:
        """Assemble a padded ``inputs_embeds``/``labels`` batch with visual tokens prepended."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> str:
        """Generate a text answer."""

    @abstractmethod
    async def generate_with_visual(self, request: VisualGenerationRequest) -> str:
        """Generate with projected visual embeddings prepended to the prompt."""
