import pytest

from lagunavision.data.adapters import HuggingFaceDatasetAdapter
from lagunavision.data.sources import DATASET_SOURCES


def test_adapter_normalizes_common_hf_row() -> None:
    source = DATASET_SOURCES[0]
    adapter = HuggingFaceDatasetAdapter(source)

    example = adapter._normalize(
        {
            "image": "image.jpg",
            "question": "What is shown?",
            "answer": "A terminal error.",
        },
        index=0,
    )

    assert example is not None
    assert example.source == source.id
    assert example.image == "image.jpg"


def test_adapter_skips_rows_without_answer() -> None:
    adapter = HuggingFaceDatasetAdapter(DATASET_SOURCES[0])

    assert adapter._normalize({"image": "image.jpg", "question": "What is shown?"}, index=0) is None
