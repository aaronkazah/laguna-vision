import asyncio
from pathlib import Path

import torch
from PIL import Image

from lagunavision.encoders.pil_patch import PilPatchVisionEncoder
from lagunavision.positions.normalized_2d import Normalized2DPositionEncoder
from lagunavision.tiling.anyres import AnyResTiler
from lagunavision.train.visual_bridge import _build_dataset, _collate
from lagunavision.types import EvalManifestItem


class _FakeBackbone:
    def tokenize_example(self, question: str, answer: str, context: str = ""):
        return torch.tensor([1, 2, 3]), torch.tensor([4, 5])


def _item(item_id: str, image: Path, term: str) -> EvalManifestItem:
    return EvalManifestItem(
        id=item_id,
        image=image,
        question="what is shown?",
        ocr_text="",
        rubric="vqa",
        must_include=(term,),
        accepted_fix_terms=(),
        must_not_include=(),
    )


def test_build_dataset_precomputes_features_and_tokens(tmp_path) -> None:
    image = tmp_path / "screen.png"
    Image.new("RGB", (256, 128), "white").save(image)
    items = (_item("a", image, "cat"), _item("b", image, "dog"))

    dataset = asyncio.run(
        _build_dataset(
            items,
            _FakeBackbone(),
            AnyResTiler(max_tiles=4),
            Normalized2DPositionEncoder(),
            PilPatchVisionEncoder(patch_px=16),
        )
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["vf"].dim() == 2
    assert sample["vf"].shape[1] == dataset.input_dim
    assert sample["prompt_ids"].tolist() == [1, 2, 3]
    assert sample["answer_ids"].tolist() == [4, 5]


def test_collate_keeps_per_sample_lists(tmp_path) -> None:
    image = tmp_path / "screen.png"
    Image.new("RGB", (256, 128), "white").save(image)
    dataset = asyncio.run(
        _build_dataset(
            (_item("a", image, "cat"), _item("b", image, "dog")),
            _FakeBackbone(),
            AnyResTiler(max_tiles=4),
            Normalized2DPositionEncoder(),
            PilPatchVisionEncoder(patch_px=16),
        )
    )

    batch = _collate([dataset[0], dataset[1]])

    assert len(batch["vf"]) == 2
    assert len(batch["prompt_ids"]) == 2
    assert len(batch["answer_ids"]) == 2
