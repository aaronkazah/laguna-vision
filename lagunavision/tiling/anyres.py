from __future__ import annotations

from dataclasses import dataclass

from lagunavision.types import CropBox, Grid, Tile

DEFAULT_GRIDS = (
    Grid(1, 1),
    Grid(1, 2),
    Grid(2, 1),
    Grid(2, 2),
    Grid(1, 3),
    Grid(3, 1),
    Grid(2, 3),
    Grid(3, 2),
    Grid(3, 3),
)


@dataclass(frozen=True)
class AnyResTiler:
    tile_px: int = 384
    max_tiles: int = 9
    include_global: bool = True
    grids: tuple[Grid, ...] = DEFAULT_GRIDS

    def choose_grid(self, width: int, height: int) -> Grid:
        self._validate_dimensions(width, height)
        aspect = width / height
        candidates = [grid for grid in self.grids if grid.tile_count <= self.max_tiles]
        if not candidates:
            raise ValueError("at least one grid must fit within max_tiles")
        best_aspect_delta = min(abs(grid.aspect_ratio - aspect) for grid in candidates)
        detailed_candidates = [
            grid
            for grid in candidates
            if abs(grid.aspect_ratio - aspect) <= best_aspect_delta + 0.1
        ]
        return max(detailed_candidates, key=lambda grid: grid.tile_count)

    def tiles_for_size(self, width: int, height: int) -> tuple[Tile, ...]:
        self._validate_dimensions(width, height)
        grid = self.choose_grid(width, height)
        tiles: list[Tile] = []
        if self.include_global:
            tiles.append(
                Tile(
                    id="global",
                    crop=CropBox(0, 0, width, height),
                    original_width=width,
                    original_height=height,
                    tile_row=0,
                    tile_col=0,
                    grid_rows=1,
                    grid_cols=1,
                    is_global=True,
                )
            )

        for row in range(grid.rows):
            for col in range(grid.cols):
                left = round(col * width / grid.cols)
                right = round((col + 1) * width / grid.cols)
                top = round(row * height / grid.rows)
                bottom = round((row + 1) * height / grid.rows)
                tiles.append(
                    Tile(
                        id=f"tile-r{row}-c{col}",
                        crop=CropBox(left, top, right, bottom),
                        original_width=width,
                        original_height=height,
                        tile_row=row,
                        tile_col=col,
                        grid_rows=grid.rows,
                        grid_cols=grid.cols,
                    )
                )
        return tuple(tiles)

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
