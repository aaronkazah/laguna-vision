from __future__ import annotations

import json
from pathlib import Path

from lagunavision.backbones.base import Backbone, GenerationRequest
from lagunavision.data.manifest import load_manifest
from lagunavision.eval.score_eval import score_answer


async def run_text_eval(
    manifest_path: Path,
    backbone: Backbone,
    output_path: Path,
    use_ocr_context: bool = False,
) -> tuple[int, int]:
    items = load_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    passed = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            answer = await backbone.generate(
                GenerationRequest(
                    question=item.question,
                    context=item.ocr_text if use_ocr_context else "",
                    max_new_tokens=32 if item.rubric == "vqa" else 256,
                )
            )
            score = score_answer(item, answer)
            passed += int(score.passed)
            handle.write(
                json.dumps(
                    {
                        "id": item.id,
                        "answer": answer,
                        "points": score.points,
                        "passed": score.passed,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    return passed, len(items)
