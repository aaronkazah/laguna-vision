from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Mapping, Protocol, Sequence


class DatasetStage(str, Enum):
    ALIGNMENT = "alignment"
    INSTRUCTION = "instruction"
    EVAL = "eval"


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int

    @property
    def tile_count(self) -> int:
        return self.rows * self.cols

    @property
    def aspect_ratio(self) -> float:
        return self.cols / self.rows


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class Tile:
    id: str
    crop: CropBox
    original_width: int
    original_height: int
    tile_row: int
    tile_col: int
    grid_rows: int
    grid_cols: int
    is_global: bool = False

    @property
    def center_x(self) -> float:
        return ((self.crop.left + self.crop.right) / 2) / self.original_width

    @property
    def center_y(self) -> float:
        return ((self.crop.top + self.crop.bottom) / 2) / self.original_height


@dataclass(frozen=True)
class PositionFeatures:
    values: tuple[float, ...]


@dataclass(frozen=True)
class DatasetSource:
    id: str
    stage: DatasetStage
    use: str
    required: bool


@dataclass(frozen=True)
class TrainingExample:
    id: str
    source: str
    image: str
    question: str
    answer: str
    stage: DatasetStage


class DatasetAdapter(Protocol):
    source: DatasetSource

    async def examples(self, limit: int | None = None) -> AsyncIterator[TrainingExample]:
        """Yield normalized public training examples asynchronously."""


@dataclass(frozen=True)
class EvalManifestItem:
    id: str
    image: Path
    question: str
    ocr_text: str
    rubric: str
    must_include: tuple[str, ...]
    accepted_fix_terms: tuple[str, ...]
    must_not_include: tuple[str, ...]
    answer: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, object], root: Path) -> "EvalManifestItem":
        def strings(name: str) -> tuple[str, ...]:
            value = row.get(name, [])
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"{name} must be a list of strings")
            return tuple(str(item) for item in value)

        id_value = row.get("id")
        image_value = row.get("image")
        question_value = row.get("question")
        if not id_value or not image_value or not question_value:
            raise ValueError("manifest rows require id, image, and question")

        return cls(
            id=str(id_value),
            image=root / str(image_value),
            question=str(question_value),
            ocr_text=str(row.get("ocr_text", "")),
            rubric=str(row.get("rubric", "bugfix")),
            must_include=strings("must_include"),
            accepted_fix_terms=strings("accepted_fix_terms"),
            must_not_include=strings("must_not_include"),
            answer=str(row.get("answer", "")),
        )


@dataclass(frozen=True)
class ScoreBreakdown:
    item_id: str
    read_key_text: bool
    identified_cause: bool
    gave_fix: bool
    violated_negative: bool

    @property
    def points(self) -> int:
        return int(self.read_key_text) + int(self.identified_cause) + int(self.gave_fix)

    @property
    def passed(self) -> bool:
        return self.points >= 2 and not self.violated_negative
