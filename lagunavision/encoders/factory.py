from __future__ import annotations

from lagunavision.encoders.base import VisionEncoder
from lagunavision.encoders.hf_vision import HfVisionEncoder
from lagunavision.encoders.pil_patch import PilPatchVisionEncoder


def build_vision_encoder(
    name: str,
    model_id: str = "",
    patch_px: int = 32,
    device: str = "auto",
) -> VisionEncoder:
    if name == "pil":
        return PilPatchVisionEncoder(patch_px=patch_px)
    if name == "hf":
        if not model_id:
            raise ValueError("model_id is required for the hf vision encoder")
        return HfVisionEncoder(model_id=model_id, device=device)
    raise ValueError(f"unknown vision encoder: {name}")
