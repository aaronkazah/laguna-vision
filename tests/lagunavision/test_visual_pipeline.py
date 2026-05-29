import asyncio

from PIL import Image

from lagunavision.tiling.anyres import AnyResTiler
from lagunavision.visual_pipeline import LagunaVisionImagePipeline


def test_visual_pipeline_tiles_real_image_size(tmp_path) -> None:
    image = tmp_path / "screen.png"
    Image.new("RGB", (800, 400), "white").save(image)
    pipeline = LagunaVisionImagePipeline(
        backbone=None,
        projector=None,
        tiler=AnyResTiler(max_tiles=4),
        encoder=None,
        positioner=None,
    )

    tiles = pipeline._tiles_for_image(image)

    assert tiles[0].is_global
    assert tiles[0].original_width == 800
    assert tiles[0].original_height == 400
