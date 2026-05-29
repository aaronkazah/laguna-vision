from lagunavision.data.sources import DATASET_SOURCES, required_sources
from lagunavision.data.spatial_ocr import generate_spatial_ocr_manifest


def test_required_dataset_sources_are_public_and_general() -> None:
    ids = {source.id for source in required_sources()}
    assert "liuhaotian/LLaVA-Pretrain" in ids
    assert "HuggingFaceM4/DocumentVQA" in ids
    assert "lmms-lab/textvqa" in ids
    assert "howard-hou/OCR-VQA" in ids
    assert "synthetic-spatial-ocr" in ids
    assert all("code" not in source.id.casefold() for source in DATASET_SOURCES)


def test_spatial_ocr_writes_manifest_and_images(tmp_path) -> None:
    examples = generate_spatial_ocr_manifest(tmp_path, count=2)

    assert len(examples) == 2
    assert (tmp_path / "manifest.jsonl").exists()
    assert all((tmp_path / example.image).exists() for example in examples)
