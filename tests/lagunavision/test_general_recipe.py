import base64
import io
import json
from pathlib import Path

from PIL import Image

from lagunavision.data import general_recipe
from lagunavision.data.general_recipe import (
    PILOT_300K_RECIPE,
    RawExample,
    _example_from_hf_row,
    _examples_from_ocr_vqa_row,
    _materialize_example_image,
    _resolve_image_path,
    materialize_general_recipe,
    recipe_summary,
)


def _line_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def test_recipe_locks_exact_general_vision_sources() -> None:
    summary = recipe_summary(PILOT_300K_RECIPE)

    assert summary["alignment_train"] == 120_000
    assert summary["instruction_train"] == 180_000
    assert summary["total_train"] == 300_000
    source_ids = {source["id"] for source in summary["sources"]}
    assert {
        "llava-pretrain-558k",
        "llava-instruct-150k",
        "sharegpt4v-100k",
        "gqa-balanced",
        "documentvqa",
        "textvqa",
        "ocr-vqa",
        "chartqa",
        "websight",
        "rico-screenqa",
        "rico-screen2words",
        "websrc",
        "spatial-ocr",
    } == source_ids


def test_train_budget_scales_recipe_proportionally() -> None:
    summary = recipe_summary(PILOT_300K_RECIPE, train_budget=100_000)

    assert summary["total_train"] == 100_000
    assert summary["alignment_train"] == 40_000
    assert summary["instruction_train"] == 60_000
    counts = {source["id"]: source["train_count"] for source in summary["sources"]}
    assert counts["llava-pretrain-558k"] == 40_000
    assert counts["llava-instruct-150k"] == 21_667
    assert counts["sharegpt4v-100k"] == 11_667
    assert counts["gqa-balanced"] == 8_333
    assert counts["rico-screen2words"] == 1_334
    assert all(count > 0 for count in counts.values())


def test_general_materialize_writes_staged_manifests_and_controls(tmp_path, monkeypatch) -> None:
    def fake_examples(source, **_kwargs):
        for index in range(source.total_count):
            image = Image.new("RGB", (16, 16), (index % 255, 0, 0))
            yield RawExample(
                id=f"{source.id}:{index}",
                image=image,
                question=f"What is shown for {source.id}?",
                answer=f"{source.id} answer {index}",
            )

    monkeypatch.setattr(general_recipe, "_iter_source_examples", fake_examples)

    result = materialize_general_recipe(tmp_path, sample_per_source=2)

    assert _line_count(result.alignment_train_manifest) == 2
    assert _line_count(result.alignment_eval_manifest) == 1
    assert _line_count(result.instruction_train_manifest) == 24
    assert _line_count(result.instruction_eval_manifest) == 12
    assert _line_count(result.wrong_manifest) == 12
    assert _line_count(result.blank_manifest) == 12
    recipe = json.loads(result.recipe_file.read_text(encoding="utf-8"))
    assert recipe["total_train"] == 26

    first_instruction = json.loads(result.instruction_train_manifest.read_text(encoding="utf-8").splitlines()[0])
    assert first_instruction["source_id"] == "llava-instruct-150k"
    assert (result.instruction_train_manifest.parent / first_instruction["image"]).exists()


def test_hf_row_normalizers_cover_ui_and_ocr_shapes() -> None:
    image = Image.new("RGB", (8, 8), "white")

    doc_source = next(source for source in general_recipe.GENERAL_VISION_300K_SOURCES if source.id == "documentvqa")
    doc = _example_from_hf_row(doc_source, {"image": image, "question": "Invoice total?", "answers": ["$12", "$12", "$13"]}, 0)
    assert doc is not None
    assert doc.answer == "$12"

    rico_source = next(source for source in general_recipe.GENERAL_VISION_300K_SOURCES if source.id == "rico-screenqa")
    rico = _example_from_hf_row(
        rico_source,
        {
            "image": image,
            "question": "What button is at the bottom?",
            "ground_truth": [{"full_answer": "The Save button is at the bottom."}],
        },
        1,
    )
    assert rico is not None
    assert rico.answer == "The Save button is at the bottom."

    screen2words_source = next(source for source in general_recipe.GENERAL_VISION_300K_SOURCES if source.id == "rico-screen2words")
    screen = _example_from_hf_row(screen2words_source, {"image": image, "captions": ["short", "a detailed settings screen"]}, 2)
    assert screen is not None
    assert screen.answer == "a detailed settings screen"


def test_ocr_vqa_row_expands_paired_questions_and_answers() -> None:
    image = Image.new("RGB", (8, 8), "white")
    source = next(source for source in general_recipe.GENERAL_VISION_300K_SOURCES if source.id == "ocr-vqa")

    examples = list(
        _examples_from_ocr_vqa_row(
            source,
            {
                "image": image,
                "image_id": "book-1",
                "questions": ["Who wrote this book?", "What is the title?"],
                "answers": ["Ada Lovelace", "Notes"],
                "ocr_tokens": ["ada", "notes"],
            },
            0,
        )
    )

    assert [example.question for example in examples] == ["Who wrote this book?", "What is the title?"]
    assert [example.answer for example in examples] == ["Ada Lovelace", "Notes"]
    assert examples[0].ocr_text == "ada notes"
    assert examples[0].id == "ocr-vqa:book-1:0"


def test_resolve_image_path_accepts_llava_parent_when_root_points_at_images(tmp_path: Path) -> None:
    source_json = tmp_path / "blip_laion_cc_sbu_558k.json"
    source_json.write_text("[]", encoding="utf-8")
    image = tmp_path / "llava_pretrain" / "00453" / "004539375.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")

    resolved = _resolve_image_path(
        "00453/004539375.jpg",
        source_json,
        [tmp_path / "llava_pretrain" / "images"],
    )

    assert resolved == image.resolve()


def test_materialize_example_image_decodes_long_inline_base64(tmp_path: Path) -> None:
    source = next(source for source in general_recipe.GENERAL_VISION_300K_SOURCES if source.id == "documentvqa")
    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    path = _materialize_example_image(encoded, tmp_path, source, "train", 0)

    assert path.exists()
    assert Image.open(path).size == (32, 32)


def test_hf_rows_retry_transient_load_dataset_errors(monkeypatch) -> None:
    class TransientHfError(Exception):
        response = type("Response", (), {"status_code": 504})()

    calls = 0
    image = Image.new("RGB", (8, 8), "white")

    def fake_load_dataset(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientHfError("504 Gateway Time-out")
        return [{"image": image, "question": "Invoice total?", "answers": ["$12"]}]

    monkeypatch.setattr(general_recipe, "_require_datasets", lambda: fake_load_dataset)
    monkeypatch.setattr(general_recipe.time, "sleep", lambda _delay: None)

    source = next(source for source in general_recipe.GENERAL_VISION_300K_SOURCES if source.id == "documentvqa")
    examples = list(general_recipe._iter_hf_rows(source))

    assert calls == 2
    assert len(examples) == 1
    assert examples[0].answer == "$12"
