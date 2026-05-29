import asyncio
import json
from contextlib import nullcontext
from pathlib import Path

from lagunavision.backbones.base import GenerationRequest
from lagunavision.eval.ablation import evaluate_arms
from lagunavision.types import EvalManifestItem


def _item(item_id: str) -> EvalManifestItem:
    return EvalManifestItem(
        id=item_id,
        image=Path(f"{item_id}.png"),
        question="what animal is shown?",
        ocr_text="the cat",
        rubric="vqa",
        must_include=("cat",),
        accepted_fix_terms=(),
        must_not_include=(),
    )


class _EchoContextBackbone:
    """generate() returns whatever context it is handed, so the OCR arm sees OCR text."""

    def adapter_disabled(self):
        return nullcontext()

    async def generate(self, request: GenerationRequest) -> str:
        return request.context


class _CannedPipeline:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.backbone = _EchoContextBackbone()

    async def answer_image(self, image, question, context="", max_new_tokens=128) -> str:
        return self._answer


def test_ablation_isolates_visual_capability(tmp_path) -> None:
    items = (_item("a"), _item("b"))
    output = tmp_path / "ablation.jsonl"

    summary = asyncio.run(
        evaluate_arms(
            items,
            backbone=_EchoContextBackbone(),
            pipeline=_CannedPipeline("a cat"),
            untrained=_CannedPipeline("nothing here"),
            output=output,
        )
    )

    assert summary["arms"]["text_only"]["pass_rate"] == 0.0
    assert summary["arms"]["ocr_only"]["pass_rate"] == 1.0
    assert summary["arms"]["image_only"]["pass_rate"] == 1.0
    assert summary["arms"]["image_untrained"]["pass_rate"] == 0.0
    assert summary["capability_delta"] == 1.0
    assert summary["connector_delta"] == 1.0
    assert summary["capability_passed"] is True

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(items) * 5
    assert {row["arm"] for row in rows} == {
        "text_only",
        "ocr_only",
        "image_only",
        "image_ocr",
        "image_untrained",
    }
    assert (output.parent / "ablation_summary.json").exists()


def test_ablation_gate_fails_without_visual_lift(tmp_path) -> None:
    items = (_item("a"),)

    summary = asyncio.run(
        evaluate_arms(
            items,
            backbone=_EchoContextBackbone(),
            pipeline=_CannedPipeline(""),
            untrained=_CannedPipeline(""),
            output=tmp_path / "ablation.jsonl",
        )
    )

    assert summary["capability_delta"] == 0.0
    assert summary["capability_passed"] is False


class _AdapterSpyBackbone:
    def __init__(self) -> None:
        self.disabled_calls = 0

    def adapter_disabled(self):
        self.disabled_calls += 1
        return nullcontext()

    async def generate(self, request: GenerationRequest) -> str:
        return request.context


class _SpyPipeline:
    def __init__(self, answer: str, backbone: _AdapterSpyBackbone) -> None:
        self._answer = answer
        self.backbone = backbone

    async def answer_image(self, image, question, context="", max_new_tokens=128) -> str:
        return self._answer


def test_untrained_arm_disables_adapter(tmp_path) -> None:
    items = (_item("a"), _item("b"))
    trained = _AdapterSpyBackbone()
    untrained = _AdapterSpyBackbone()

    asyncio.run(
        evaluate_arms(
            items,
            backbone=_EchoContextBackbone(),
            pipeline=_SpyPipeline("a cat", trained),
            untrained=_SpyPipeline("a cat", untrained),
            output=tmp_path / "ablation.jsonl",
        )
    )

    # the untrained control disables the adapter once per item; trained arms never do
    assert untrained.disabled_calls == len(items)
    assert trained.disabled_calls == 0
