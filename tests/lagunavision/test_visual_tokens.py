import asyncio

import pytest

from lagunavision.encoders.pil_patch import PilPatchVisionEncoder
from lagunavision.encoders.base import EncodedTile
from lagunavision.positions.normalized_2d import Normalized2DPositionEncoder
from lagunavision.projectors.mlp import MlpProjector
from lagunavision.projectors.resampler import ResamplerProjector
from lagunavision.tiling.anyres import AnyResTiler

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")


def test_visual_encoder_and_projector_emit_trainable_embeddings(tmp_path) -> None:
    asyncio.run(_assert_visual_tokens(tmp_path))


async def _assert_visual_tokens(tmp_path) -> None:
    image_path = tmp_path / "screenshot.png"
    Image.new("RGB", (640, 480), "white").save(image_path)
    tiles = AnyResTiler(max_tiles=4).tiles_for_size(640, 480)
    encoded = await PilPatchVisionEncoder().encode(image_path, tiles)
    positions = Normalized2DPositionEncoder().encode_tiles(tiles)
    projector = MlpProjector(
        input_dim=encoded[0].feature_dim + len(positions[0].values),
        embedding_dim=128,
    )

    projected = await projector.project(encoded, positions)

    assert projected.embeddings.shape == (len(tiles), 128)
    assert projected.embeddings.requires_grad


def test_projector_keeps_dense_patch_tokens(tmp_path) -> None:
    tiles = AnyResTiler(max_tiles=1).tiles_for_size(640, 480)
    encoded = dense_tiles(tiles)
    positions = Normalized2DPositionEncoder().encode_tiles(tiles)
    projector = MlpProjector(
        input_dim=encoded[0].feature_dim + len(positions[0].values),
        embedding_dim=128,
    )

    projected = asyncio.run(projector.project(encoded, positions))

    assert projected.embeddings.shape == (len(tiles) * 3, 128)


def test_resampler_projector_emits_fixed_visual_token_count() -> None:
    tiles = AnyResTiler(max_tiles=1).tiles_for_size(640, 480)
    encoded = dense_tiles(tiles)
    positions = Normalized2DPositionEncoder().encode_tiles(tiles)
    projector = ResamplerProjector(
        input_dim=encoded[0].feature_dim + len(positions[0].values),
        embedding_dim=128,
        visual_tokens=5,
    )

    projected = asyncio.run(projector.project(encoded, positions))

    assert projected.embeddings.shape == (5, 128)
    assert projected.embeddings.requires_grad


def dense_tiles(tiles):
    return tuple(
        EncodedTile(
            tile=tile,
            patch_count=3,
            feature_dim=4,
            features=torch.ones(3, 4),
        )
        for tile in tiles
    )
