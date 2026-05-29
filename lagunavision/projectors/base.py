from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from lagunavision.encoders.base import EncodedTile
from lagunavision.types import PositionFeatures


@dataclass(frozen=True)
class ProjectedVisualTokens:
    token_count: int
    embedding_dim: int
    embeddings: Any = None


class Projector(Protocol):
    async def project(
        self,
        encoded_tiles: tuple[EncodedTile, ...],
        positions: tuple[PositionFeatures, ...],
    ) -> ProjectedVisualTokens:
        """Project visual features into text-backbone embedding space."""
