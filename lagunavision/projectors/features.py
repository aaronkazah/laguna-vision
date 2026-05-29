from __future__ import annotations

from typing import Any

from lagunavision.encoders.base import EncodedTile
from lagunavision.types import PositionFeatures


def stack_visual_features(
    encoded_tiles: tuple[EncodedTile, ...],
    positions: tuple[PositionFeatures, ...],
    device: Any,
):
    import torch

    if len(encoded_tiles) != len(positions):
        raise ValueError("encoded_tiles and positions must have matching lengths")
    rows = []
    for encoded, position in zip(encoded_tiles, positions):
        if encoded.features is None:
            raise ValueError("encoded tiles must include tensor features")
        feature = encoded.features.reshape(-1, encoded.feature_dim).to(device)
        position_tensor = torch.tensor(position.values, dtype=feature.dtype, device=feature.device).repeat(
            feature.shape[0], 1
        )
        rows.append(torch.cat([feature, position_tensor], dim=1))
    return torch.cat(rows, dim=0)
