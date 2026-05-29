from lagunavision.positions.normalized_2d import Normalized2DPositionEncoder
from lagunavision.tiling.anyres import AnyResTiler


def test_position_features_are_stable_and_normalized() -> None:
    tiles = AnyResTiler().tiles_for_size(width=1200, height=800)
    features = Normalized2DPositionEncoder().encode_tiles(tiles)

    assert len(features) == len(tiles)
    assert all(len(feature.values) == 7 for feature in features)
    assert all(0.0 <= value <= 1.0 for feature in features for value in feature.values[:4])

