from __future__ import annotations

from dataclasses import dataclass

from lagunavision.encoders.base import EncodedTile
from lagunavision.projectors.base import ProjectedVisualTokens
from lagunavision.projectors.features import stack_visual_features
from lagunavision.types import PositionFeatures


@dataclass
class MlpProjector:
    input_dim: int
    embedding_dim: int
    hidden_dim: int = 256

    def __post_init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install torch to use MlpProjector.") from exc

        self.module = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.embedding_dim),
        )

    async def project(
        self,
        encoded_tiles: tuple[EncodedTile, ...],
        positions: tuple[PositionFeatures, ...],
    ) -> ProjectedVisualTokens:
        device = next(self.module.parameters()).device
        visual_features = stack_visual_features(encoded_tiles, positions, device)
        embeddings = self.module(visual_features)
        return ProjectedVisualTokens(token_count=embeddings.shape[0], embedding_dim=embeddings.shape[1], embeddings=embeddings)
