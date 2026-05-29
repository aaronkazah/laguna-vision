from __future__ import annotations

from dataclasses import dataclass

from lagunavision.types import PositionFeatures, Tile


@dataclass(frozen=True)
class Normalized2DPositionEncoder:
    def encode_tile(self, tile: Tile) -> PositionFeatures:
        values = (
            self._unit(tile.center_x),
            self._unit(tile.center_y),
            self._index(tile.tile_row, tile.grid_rows),
            self._index(tile.tile_col, tile.grid_cols),
            float(tile.grid_rows),
            float(tile.grid_cols),
            1.0 if tile.is_global else 0.0,
        )
        return PositionFeatures(values=values)

    def encode_tiles(self, tiles: tuple[Tile, ...]) -> tuple[PositionFeatures, ...]:
        return tuple(self.encode_tile(tile) for tile in tiles)

    def _index(self, value: int, size: int) -> float:
        if size <= 1:
            return 0.0
        return self._unit(value / (size - 1))

    def _unit(self, value: float) -> float:
        return min(1.0, max(0.0, value))

