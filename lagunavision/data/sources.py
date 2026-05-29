from __future__ import annotations

from lagunavision.types import DatasetSource, DatasetStage

DATASET_SOURCES = (
    DatasetSource(
        id="liuhaotian/LLaVA-Pretrain",
        stage=DatasetStage.ALIGNMENT,
        use="LLaVA-compatible alignment data when raw image assets are present.",
        required=False,
    ),
    DatasetSource(
        id="HuggingFaceM4/DocumentVQA",
        stage=DatasetStage.INSTRUCTION,
        use="Document and form understanding.",
        required=True,
    ),
    DatasetSource(
        id="lmms-lab/textvqa",
        stage=DatasetStage.INSTRUCTION,
        use="Natural images with embedded text.",
        required=True,
    ),
    DatasetSource(
        id="howard-hou/OCR-VQA",
        stage=DatasetStage.INSTRUCTION,
        use="OCR-heavy visual QA.",
        required=True,
    ),
    DatasetSource(
        id="synthetic-spatial-ocr",
        stage=DatasetStage.INSTRUCTION,
        use="Generated spatial grounding examples.",
        required=False,
    ),
    DatasetSource(
        id="HuggingFaceM4/ChartQA",
        stage=DatasetStage.INSTRUCTION,
        use="Chart and diagram understanding.",
        required=True,
    ),
    DatasetSource(
        id="HuggingFaceM4/FineVision",
        stage=DatasetStage.INSTRUCTION,
        use="Broad multimodal instruction mixture.",
        required=False,
    ),
)


def required_sources() -> tuple[DatasetSource, ...]:
    return tuple(source for source in DATASET_SOURCES if source.required)
