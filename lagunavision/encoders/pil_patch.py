from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from lagunavision.encoders.base import EncodedTile
from lagunavision.types import Tile


@dataclass(frozen=True)
class PilPatchVisionEncoder:
    patch_px: int = 32

    async def encode(self, image: Path, tiles: tuple[Tile, ...]) -> tuple[EncodedTile, ...]:
        return await asyncio.to_thread(self._encode_sync, image, tiles)

    def _encode_sync(self, image: Path, tiles: tuple[Tile, ...]) -> tuple[EncodedTile, ...]:
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install runtime dependencies with `python -m pip install -e '.[dev]' 'torch>=2.3'`.") from exc

        source = Image.open(image).convert("RGB")
        encoded: list[EncodedTile] = []
        for tile in tiles:
            crop = source.crop((tile.crop.left, tile.crop.top, tile.crop.right, tile.crop.bottom))
            crop = crop.resize((self.patch_px, self.patch_px))
            pixels = torch.tensor(list(crop.getdata()), dtype=torch.float32) / 255.0
            mean = pixels.mean(dim=0)
            std = pixels.std(dim=0)
            features = torch.cat(
                [
                    mean,
                    std,
                    torch.tensor(
                        [
                            tile.center_x,
                            tile.center_y,
                            tile.tile_row,
                            tile.tile_col,
                            tile.grid_rows,
                            tile.grid_cols,
                            1.0 if tile.is_global else 0.0,
                        ],
                        dtype=torch.float32,
                    ),
                ]
            ).unsqueeze(0)
            encoded.append(EncodedTile(tile=tile, patch_count=1, feature_dim=features.shape[-1], features=features))
        return tuple(encoded)
