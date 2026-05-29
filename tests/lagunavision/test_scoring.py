from pathlib import Path

from lagunavision.eval.score_eval import score_answer
from lagunavision.types import EvalManifestItem


def test_score_passes_when_answer_reads_cause_and_fix() -> None:
    item = EvalManifestItem(
        id="py_missing_package_001",
        image=Path("images/001.png"),
        question="What is wrong?",
        ocr_text="ModuleNotFoundError: No module named requests",
        rubric="bugfix",
        must_include=("ModuleNotFoundError", "requests"),
        accepted_fix_terms=("pip install requests",),
        must_not_include=("syntax error",),
    )

    score = score_answer(
        item,
        "The screenshot shows ModuleNotFoundError for requests because the package is missing. Run pip install requests.",
    )

    assert score.points == 3
    assert score.passed


def test_score_fails_on_negative_term() -> None:
    item = EvalManifestItem(
        id="bad",
        image=Path("images/001.png"),
        question="What is wrong?",
        ocr_text="ModuleNotFoundError: No module named requests",
        rubric="bugfix",
        must_include=("ModuleNotFoundError",),
        accepted_fix_terms=("pip install requests",),
        must_not_include=("syntax error",),
    )

    score = score_answer(item, "ModuleNotFoundError because missing. pip install requests. It is a syntax error.")

    assert score.violated_negative
    assert not score.passed


def test_description_rubric_scores_page_explanation() -> None:
    item = EvalManifestItem(
        id="web_example",
        image=Path("images/001.png"),
        question="What is this page?",
        ocr_text="Example Domain",
        rubric="description",
        must_include=("Example Domain",),
        accepted_fix_terms=("domain", "examples"),
        must_not_include=(),
    )

    score = score_answer(item, "This is the Example Domain page used for illustrative examples.")

    assert score.passed


def test_vqa_rubric_scores_exact_answer() -> None:
    item = EvalManifestItem(
        id="vqa",
        image=Path("images/001.png"),
        question="What brand?",
        ocr_text="",
        rubric="vqa",
        must_include=("nokia",),
        accepted_fix_terms=("nokia",),
        must_not_include=(),
    )

    assert score_answer(item, "The answer is Nokia.").passed
