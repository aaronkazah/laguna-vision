from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from lagunavision.data.sources import DATASET_SOURCES
from lagunavision.types import DatasetSource, DatasetStage, TrainingExample


@dataclass(frozen=True)
class HuggingFaceDatasetAdapter:
    source: DatasetSource
    split: str = "train"

    async def examples(self, limit: int | None = None) -> AsyncIterator[TrainingExample]:
        rows = await asyncio.to_thread(self._load_rows)
        emitted = 0
        for row in rows:
            normalized = self._normalize(row, emitted)
            if normalized is not None:
                yield normalized
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def _load_rows(self) -> Any:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install dataset dependencies with `python -m pip install -e '.[data]'`.") from exc
        return load_dataset(self.source.id, split=self.split, streaming=True)

    def _normalize(self, row: Mapping[str, object], index: int) -> TrainingExample | None:
        image = _first_string(row, ("image", "image_path", "url"))
        question = _first_string(row, ("question", "prompt", "text", "caption"))
        answer = _first_string(row, ("answer", "answers", "label", "caption"))
        if not image or not question or not answer:
            return None
        return TrainingExample(
            id=f"{self.source.id}:{self.split}:{index}",
            source=self.source.id,
            image=image,
            question=question,
            answer=answer,
            stage=self.source.stage,
        )


def required_hf_adapters() -> tuple[HuggingFaceDatasetAdapter, ...]:
    return tuple(
        HuggingFaceDatasetAdapter(source)
        for source in DATASET_SOURCES
        if source.required and source.id != "synthetic-spatial-ocr"
    )


def _first_string(row: Mapping[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return ""
