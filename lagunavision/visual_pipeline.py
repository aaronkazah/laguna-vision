from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lagunavision.backbones.base import Backbone, VisualGenerationRequest
from lagunavision.backbones.factory import build_backbone
from lagunavision.encoders.base import VisionEncoder
from lagunavision.encoders.factory import build_vision_encoder
from lagunavision.positions.normalized_2d import Normalized2DPositionEncoder
from lagunavision.projectors.mlp import MlpProjector
from lagunavision.projectors.resampler import ResamplerProjector
from lagunavision.tiling.anyres import AnyResTiler


@dataclass(frozen=True)
class VisualProjectorSpec:
    input_dim: int
    embedding_dim: int
    hidden_dim: int = 256
    projector: str = "mlp"
    visual_tokens: int = 64
    encoder: str = "pil"
    encoder_id: str = ""
    max_tiles: int = 4
    patch_px: int = 32


def build_projector(spec: VisualProjectorSpec) -> MlpProjector | ResamplerProjector:
    if spec.projector == "mlp":
        return MlpProjector(
            input_dim=spec.input_dim,
            embedding_dim=spec.embedding_dim,
            hidden_dim=spec.hidden_dim,
        )
    if spec.projector == "resampler":
        return ResamplerProjector(
            input_dim=spec.input_dim,
            embedding_dim=spec.embedding_dim,
            hidden_dim=spec.hidden_dim,
            visual_tokens=spec.visual_tokens,
        )
    raise ValueError(f"unknown projector: {spec.projector}")


@dataclass
class LagunaVisionImagePipeline:
    backbone: Backbone
    projector: MlpProjector
    tiler: AnyResTiler
    encoder: VisionEncoder
    positioner: Normalized2DPositionEncoder

    @classmethod
    async def from_checkpoint(
        cls,
        checkpoint: Path,
        spec: VisualProjectorSpec,
        backbone_name: str = "laguna",
        model_id: str = "",
        device: str = "auto",
        vision_device: str = "auto",
        lora_dir: Path | None = None,
    ) -> "LagunaVisionImagePipeline":
        import torch

        backbone = build_backbone(backbone_name, model_id, device=device)
        await backbone.load()
        if lora_dir is not None:
            backbone.load_adapter(lora_dir)
        projector = build_projector(spec)
        projector.module.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        projector.module.to(backbone.resolved_device)
        projector.module.eval()
        return cls(
            backbone=backbone,
            projector=projector,
            tiler=AnyResTiler(max_tiles=spec.max_tiles),
            encoder=build_vision_encoder(spec.encoder, spec.encoder_id, spec.patch_px, vision_device),
            positioner=Normalized2DPositionEncoder(),
        )

    async def answer_image(self, image: Path, question: str, context: str = "", max_new_tokens: int = 128) -> str:
        tiles = self._tiles_for_image(image)
        encoded = await self.encoder.encode(image, tiles)
        positions = self.positioner.encode_tiles(tiles)
        projected = await self.projector.project(encoded, positions)
        return await self.backbone.generate_with_visual(
            VisualGenerationRequest(
                question=question,
                context=context,
                visual_embeddings=projected.embeddings,
                max_new_tokens=max_new_tokens,
            )
        )

    def _tiles_for_image(self, image: Path):
        from PIL import Image

        with Image.open(image) as opened:
            width, height = opened.size
        return self.tiler.tiles_for_size(width, height)
