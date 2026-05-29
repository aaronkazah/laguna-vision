from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lagunavision.backbones.base import Backbone, GenerationRequest
from lagunavision.data.manifest import load_manifest
from lagunavision.eval.score_eval import score_answer
from lagunavision.types import EvalManifestItem
from lagunavision.visual_pipeline import (
    LagunaVisionImagePipeline,
    VisualProjectorSpec,
    build_projector,
)

DEFAULT_CAPABILITY_THRESHOLD = 0.15


class ImageAnswerer(Protocol):
    backbone: Backbone

    async def answer_image(
        self, image: Path, question: str, context: str = "", max_new_tokens: int = 128
    ) -> str: ...


def _budget(item: EvalManifestItem) -> int:
    """One token budget per item across every arm, so arms stay comparable."""
    return 32 if item.rubric == "vqa" else 256


class EvalArm(ABC):
    """One leakage-controlled arm: how a single input combination answers an item.

    ``use_ocr`` toggles whether detected OCR text leaks into the prompt, so the
    same arm class covers both the blind and OCR-fed variants.
    """

    def __init__(self, name: str, *, use_ocr: bool) -> None:
        self.name = name
        self._use_ocr = use_ocr

    def _context(self, item: EvalManifestItem) -> str:
        return item.ocr_text if self._use_ocr else ""

    @abstractmethod
    async def answer(self, item: EvalManifestItem) -> str: ...


class TextArm(EvalArm):
    """Text-only backbone arm: isolates the blind prior, or what plain OCR adds."""

    def __init__(self, name: str, backbone: Backbone, *, use_ocr: bool) -> None:
        super().__init__(name, use_ocr=use_ocr)
        self._backbone = backbone

    async def answer(self, item: EvalManifestItem) -> str:
        return await self._backbone.generate(
            GenerationRequest(question=item.question, context=self._context(item), max_new_tokens=_budget(item))
        )


class ImageArm(EvalArm):
    """Visual arm: feeds the image through a pipeline's trained or untrained connector.

    ``disable_adapter`` turns off any trained LoRA adapter on the backbone for the
    duration of the call, so the untrained control reflects the pre-training state
    rather than inheriting the adapted weights.
    """

    def __init__(
        self, name: str, pipeline: ImageAnswerer, *, use_ocr: bool, disable_adapter: bool = False
    ) -> None:
        super().__init__(name, use_ocr=use_ocr)
        self._pipeline = pipeline
        self._disable_adapter = disable_adapter

    async def answer(self, item: EvalManifestItem) -> str:
        guard = self._pipeline.backbone.adapter_disabled() if self._disable_adapter else nullcontext()
        with guard:
            return await self._pipeline.answer_image(
                item.image, item.question, context=self._context(item), max_new_tokens=_budget(item)
            )


@dataclass(frozen=True)
class ArmReport:
    arm: str
    passed: int
    total: int
    mean_points: float

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


async def _score_arm(
    arm: EvalArm, items: tuple[EvalManifestItem, ...]
) -> tuple[ArmReport, list[dict]]:
    passed = 0
    points = 0
    rows: list[dict] = []
    for item in items:
        answer = await arm.answer(item)
        score = score_answer(item, answer)
        passed += int(score.passed)
        points += score.points
        rows.append(
            {
                "arm": arm.name,
                "id": item.id,
                "answer": answer,
                "points": score.points,
                "passed": score.passed,
            }
        )
    total = len(items)
    return ArmReport(arm.name, passed, total, points / total if total else 0.0), rows


async def evaluate_arms(
    items: tuple[EvalManifestItem, ...],
    *,
    backbone: Backbone,
    pipeline: ImageAnswerer,
    untrained: ImageAnswerer,
    output: Path,
    capability_threshold: float = DEFAULT_CAPABILITY_THRESHOLD,
) -> dict:
    """Run every arm, write per-item rows, and return the capability summary.

    The visual capability metric is ``image_only - text_only``: it isolates what
    the trained system adds over the backbone's blind text prior. The
    ``image_untrained`` arm reuses the same backbone with a randomly initialized
    connector and any trained adapter disabled, so ``image_only - image_untrained``
    proves the trained weights, not the mere presence of prepended tokens, carry
    the signal.
    """
    arms: list[EvalArm] = [
        TextArm("text_only", backbone, use_ocr=False),
        TextArm("ocr_only", backbone, use_ocr=True),
        ImageArm("image_only", pipeline, use_ocr=False),
        ImageArm("image_ocr", pipeline, use_ocr=True),
        ImageArm("image_untrained", untrained, use_ocr=False, disable_adapter=True),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    reports: list[ArmReport] = []
    with output.open("w", encoding="utf-8") as handle:
        for arm in arms:
            report, rows = await _score_arm(arm, items)
            reports.append(report)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    rate = {report.arm: report.pass_rate for report in reports}
    capability_delta = rate["image_only"] - rate["text_only"]
    connector_delta = rate["image_only"] - rate["image_untrained"]
    summary = {
        "arms": {
            report.arm: {
                "passed": report.passed,
                "total": report.total,
                "pass_rate": report.pass_rate,
                "mean_points": report.mean_points,
            }
            for report in reports
        },
        "capability_delta": capability_delta,
        "connector_delta": connector_delta,
        "capability_threshold": capability_threshold,
        "capability_passed": capability_delta >= capability_threshold,
        "items": len(items),
    }
    (output.parent / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass(frozen=True)
class AblationConfig:
    manifest: Path
    checkpoint: Path
    output: Path
    spec: VisualProjectorSpec
    backbone_name: str = "laguna"
    model_id: str = ""
    device: str = "auto"
    vision_device: str = "auto"
    limit: int = 0
    capability_threshold: float = DEFAULT_CAPABILITY_THRESHOLD
    lora_dir: Path | None = None


async def run_ablation(config: AblationConfig) -> dict:
    items = load_manifest(config.manifest)
    if config.limit > 0:
        items = items[: config.limit]
    if not items:
        raise ValueError("manifest has no items")

    pipeline = await LagunaVisionImagePipeline.from_checkpoint(
        checkpoint=config.checkpoint,
        spec=config.spec,
        backbone_name=config.backbone_name,
        model_id=config.model_id,
        device=config.device,
        vision_device=config.vision_device,
        lora_dir=config.lora_dir,
    )
    untrained = _untrained_baseline(pipeline, config.spec)
    return await evaluate_arms(
        items,
        backbone=pipeline.backbone,
        pipeline=pipeline,
        untrained=untrained,
        output=config.output,
        capability_threshold=config.capability_threshold,
    )


def _untrained_baseline(
    pipeline: LagunaVisionImagePipeline, spec: VisualProjectorSpec
) -> LagunaVisionImagePipeline:
    """Same backbone and encoder, fresh random connector — the no-training control."""
    projector = build_projector(spec)
    projector.module.to(pipeline.backbone.resolved_device)
    projector.module.eval()
    return dataclasses.replace(pipeline, projector=projector)
