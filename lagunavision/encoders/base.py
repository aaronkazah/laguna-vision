from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lagunavision.types import Tile


@dataclass(frozen=True)
class EncodedTile:
    tile: Tile
    patch_count: int
    feature_dim: int
    features: Any = None


class VisionEncoder(Protocol):
    async def encode(self, image: Path, tiles: tuple[Tile, ...]) -> tuple[EncodedTile, ...]:
        """Encode image tiles asynchronously."""
