from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lagunavision.defaults import DEFAULT_VISION_TOWER
from lagunavision.devices import resolve_torch_device
from lagunavision.encoders.base import EncodedTile
from lagunavision.types import Tile


@dataclass
class HfVisionEncoder:
    model_id: str = DEFAULT_VISION_TOWER
    device: str = "auto"

    def __post_init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None

    async def encode(self, image: Path, tiles: tuple[Tile, ...]) -> tuple[EncodedTile, ...]:
        await self._ensure_loaded()
        return await asyncio.to_thread(self._encode_sync, image, tiles)

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install Llama dependencies with `python -m pip install -e '.[llama]'`.") from exc

        resolved_device = resolve_torch_device(self.device)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.eval()
        self._model.to(resolved_device)
        self.device = resolved_device
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _encode_sync(self, image: Path, tiles: tuple[Tile, ...]) -> tuple[EncodedTile, ...]:
        if self._processor is None or self._model is None:
            raise RuntimeError("vision encoder is not loaded")

        import torch
        from PIL import Image

        source = Image.open(image).convert("RGB")
        crops = [
            source.crop((tile.crop.left, tile.crop.top, tile.crop.right, tile.crop.bottom))
            for tile in tiles
        ]
        inputs = self._processor(images=crops, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(self._model, "vision_model") and "pixel_values" in inputs:
                outputs = self._model.vision_model(pixel_values=inputs["pixel_values"])
                features = outputs.last_hidden_state
            elif hasattr(self._model, "get_image_features"):
                features = self._model.get_image_features(**inputs)
            else:
                outputs = self._model(**inputs)
                features = getattr(outputs, "last_hidden_state", None)
                if features is None:
                    features = getattr(outputs, "pooler_output", None)
                if features is None:
                    features = getattr(outputs, "image_embeds", None)
                if features is None:
                    raise RuntimeError(f"{self.model_id} did not expose image features")

        encoded: list[EncodedTile] = []
        for index, tile in enumerate(tiles):
            tile_features = features[index]
            if tile_features.ndim == 1:
                tile_features = tile_features.unsqueeze(0)
            tile_features = tile_features.detach().cpu().float()
            encoded.append(
                EncodedTile(
                    tile=tile,
                    patch_count=tile_features.shape[0],
                    feature_dim=tile_features.shape[-1],
                    features=tile_features,
                )
            )
        return tuple(encoded)
