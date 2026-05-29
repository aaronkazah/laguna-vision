from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from lagunavision.encoders.base import EncodedTile
from lagunavision.projectors.base import ProjectedVisualTokens
from lagunavision.projectors.features import stack_visual_features
from lagunavision.types import PositionFeatures


@dataclass
class ResamplerProjector:
    input_dim: int
    embedding_dim: int
    hidden_dim: int = 256
    visual_tokens: int = 64

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install torch to use ResamplerProjector.") from exc

        self.module = _build_resampler_module(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            visual_tokens=self.visual_tokens,
        )

    async def project(
        self,
        encoded_tiles: tuple[EncodedTile, ...],
        positions: tuple[PositionFeatures, ...],
    ) -> ProjectedVisualTokens:
        device = next(self.module.parameters()).device
        visual_features = stack_visual_features(encoded_tiles, positions, device)
        embeddings = self.module(visual_features)
        return ProjectedVisualTokens(
            token_count=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            embeddings=embeddings,
        )


def _build_resampler_module(input_dim: int, hidden_dim: int, embedding_dim: int, visual_tokens: int):
    """Lazily build the resampler ``nn.Module`` (torch stays an optional import)."""
    import torch

    if visual_tokens <= 0:
        raise ValueError("visual_tokens must be positive")

    class ResamplerModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(input_dim, hidden_dim)
            self.norm = torch.nn.LayerNorm(hidden_dim)
            self.output = torch.nn.Linear(hidden_dim, embedding_dim)
            self.queries = torch.nn.Parameter(torch.empty(visual_tokens, hidden_dim))
            torch.nn.init.normal_(self.queries, std=0.02)
            self._scale = sqrt(hidden_dim)

        def forward(self, visual_features):
            hidden = self.norm(self.input(visual_features))
            queries = self.queries.to(dtype=hidden.dtype)
            attention = torch.softmax(queries @ hidden.transpose(-2, -1) / self._scale, dim=-1)
            pooled = attention @ hidden
            return self.output(self.norm(pooled + queries))

    return ResamplerModule()
