from __future__ import annotations

from dataclasses import dataclass

from lagunavision.backbones.base import Backbone, GenerationRequest


@dataclass(frozen=True)
class LagunaVisionTextPipeline:
    backbone: Backbone

    async def answer(self, question: str, extracted_context: str = "") -> str:
        return await self.backbone.generate(
            GenerationRequest(question=question, context=extracted_context)
        )

