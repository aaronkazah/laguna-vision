from PIL import Image

from lagunavision.data.hf_materialize import _normalize_row, _save_image


def test_normalizes_textvqa_row() -> None:
    row = {
        "image": Image.new("RGB", (10, 10)),
        "question": "what brand?",
        "answers": ["nokia"],
        "ocr_tokens": ["NOKIA"],
    }

    normalized = _normalize_row("lmms-lab/textvqa", row, 0)

    assert normalized is not None
    assert normalized["question"] == "what brand?"
    assert normalized["answer"] == "nokia"
    assert normalized["ocr_text"] == "NOKIA"


def test_textvqa_uses_majority_answer() -> None:
    row = {
        "image": Image.new("RGB", (10, 10)),
        "question": "what brand?",
        "answers": ["toshiba", "nokia", "nokia"],
        "ocr_tokens": ["NOKIA"],
    }

    normalized = _normalize_row("lmms-lab/textvqa", row, 0)

    assert normalized is not None
    assert normalized["answer"] == "nokia"


def test_save_image_writes_png(tmp_path) -> None:
    path = _save_image(tmp_path, "train", 0, Image.new("RGB", (10, 10)))

    assert path.exists()
    assert path.suffix == ".png"
