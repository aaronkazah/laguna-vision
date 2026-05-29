from __future__ import annotations

from lagunavision.types import DatasetSource, DatasetStage

DATASET_SOURCES = (
    DatasetSource(
        id="liuhaotian/LLaVA-Pretrain",
        stage=DatasetStage.ALIGNMENT,
        use="Run 1 bridge alignment with public image-caption pairs.",
        required=True,
    ),
    DatasetSource(
        id="HuggingFaceM4/DocumentVQA",
        stage=DatasetStage.INSTRUCTION,
        use="Run 2 document text-reading instruction data.",
        required=True,
    ),
    DatasetSource(
        id="lmms-lab/textvqa",
        stage=DatasetStage.INSTRUCTION,
        use="Run 2 natural-image embedded-text QA.",
        required=True,
    ),
    DatasetSource(
        id="howard-hou/OCR-VQA",
        stage=DatasetStage.INSTRUCTION,
        use="Run 2 OCR-heavy visual QA.",
        required=True,
    ),
    DatasetSource(
        id="synthetic-spatial-ocr",
        stage=DatasetStage.INSTRUCTION,
        use="Generated top/bottom/left/right and pane-layout grounding examples.",
        required=True,
    ),
    DatasetSource(
        id="HuggingFaceM4/ChartQA",
        stage=DatasetStage.INSTRUCTION,
        use="Optional chart/diagram transfer data.",
        required=False,
    ),
    DatasetSource(
        id="HuggingFaceM4/FineVision",
        stage=DatasetStage.INSTRUCTION,
        use="Optional broad multimodal mixture if integration is fast.",
        required=False,
    ),
)


def required_sources() -> tuple[DatasetSource, ...]:
    return tuple(source for source in DATASET_SOURCES if source.required)
