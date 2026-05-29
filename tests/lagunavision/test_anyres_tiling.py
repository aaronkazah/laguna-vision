from lagunavision.tiling.anyres import AnyResTiler


def test_wide_image_chooses_wide_grid() -> None:
    grid = AnyResTiler().choose_grid(width=1920, height=540)
    assert grid.cols > grid.rows


def test_tall_image_chooses_tall_grid() -> None:
    grid = AnyResTiler().choose_grid(width=540, height=1920)
    assert grid.rows > grid.cols


def test_global_tile_is_first() -> None:
    tiles = AnyResTiler().tiles_for_size(width=1024, height=768)
    assert tiles[0].id == "global"
    assert tiles[0].is_global
    assert tiles[0].crop.width == 1024
    assert tiles[0].crop.height == 768


def test_full_hd_uses_six_tile_grid_for_detail() -> None:
    grid = AnyResTiler(max_tiles=9).choose_grid(width=1920, height=1080)

    assert (grid.rows, grid.cols) == (2, 3)
