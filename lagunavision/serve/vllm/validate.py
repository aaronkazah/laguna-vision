from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lagunavision.data.manifest import load_manifest
from lagunavision.eval.endpoint_eval import _extract_answer, _image_data_uri, _post_json
from lagunavision.eval.score_eval import score_answer
from lagunavision.serve.vllm.embedder import load_checkpoint_projector_spec
from lagunavision.serve.vllm.service import LagunaVisionVllmService
from lagunavision.serve.vllm.types import GenerationInputs, LagunaVisionVllmConfig
from lagunavision.types import EvalManifestItem, ScoreBreakdown


async def validate_vllm_against_hf(
    *,
    checkpoint: Path | str,
    manifest: Path,
    output: Path,
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "",
    limit: int | None = None,
    device: str = "auto",
    vision_device: str = "auto",
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    from lagunavision.eval.endpoint_eval import _extract_final_answer
    from lagunavision.visual_pipeline import LagunaVisionImagePipeline

    projector_path, spec, checkpoint_row = load_checkpoint_projector_spec(checkpoint)
    lora_dir = projector_path.parent / "lora"
    hf_pipeline = await LagunaVisionImagePipeline.from_checkpoint(
        checkpoint=projector_path,
        spec=spec,
        backbone_name=str(checkpoint_row.get("backbone", "laguna")),
        model_id=str(checkpoint_row.get("model_id", "")),
        device=device,
        vision_device=vision_device,
        lora_dir=lora_dir if lora_dir.exists() else None,
    )
    service = LagunaVisionVllmService.from_config(
        LagunaVisionVllmConfig(
            checkpoint=checkpoint,
            vllm_base_url=vllm_base_url,
            model=model,
            vision_device=vision_device,
            max_new_tokens=max_new_tokens,
            allow_local_files=True,
        )
    )

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(load_manifest(manifest)):
        if limit is not None and index >= limit:
            break
        hf_raw = await hf_pipeline.answer_image(
            image=item.image,
            question=item.question,
            context=item.ocr_text,
            max_new_tokens=max_new_tokens,
        )
        vllm_response = await service.answer_inputs(
            GenerationInputs(
                image=str(item.image),
                question=item.question,
                context=item.ocr_text,
                max_new_tokens=max_new_tokens,
            )
        )
        hf_answer = _extract_final_answer(str(hf_raw))
        vllm_answer = _extract_final_answer(str(vllm_response.get("answer", "")))
        hf_score = score_answer(item, hf_answer)
        vllm_score = score_answer(item, vllm_answer)
        rows.append(
            _comparison_row(
                item=item,
                hf_answer=hf_answer,
                vllm_answer=vllm_answer,
                hf_raw=hf_raw,
                vllm_raw=vllm_response,
                hf_score=hf_score,
                vllm_score=vllm_score,
                index=index,
            )
        )

    _write_jsonl(output, rows)
    return _summary(rows, output)


def compare_vllm_gateway_with_hf_endpoint(
    *,
    hf_endpoint: str,
    vllm_gateway_url: str,
    manifest: Path,
    output: Path,
    hf_token: str = "",
    vllm_token: str = "",
    limit: int | None = None,
    max_new_tokens: int = 32,
    timeout: float = 120.0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    gateway_endpoint = _gateway_generate_endpoint(vllm_gateway_url)
    for index, item in enumerate(load_manifest(manifest)):
        if limit is not None and index >= limit:
            break
        payload = {
            "inputs": {
                "image": _image_data_uri(item.image),
                "question": item.question,
                "context": item.ocr_text,
                "max_new_tokens": max_new_tokens,
            }
        }
        hf_raw = _post_json(hf_endpoint, payload, token=hf_token, timeout=timeout)
        vllm_raw = _post_json(gateway_endpoint, payload, token=vllm_token, timeout=timeout)
        hf_answer = _extract_answer(hf_raw)
        vllm_answer = _extract_answer(vllm_raw)
        hf_score = score_answer(item, hf_answer)
        vllm_score = score_answer(item, vllm_answer)
        rows.append(
            _comparison_row(
                item=item,
                hf_answer=hf_answer,
                vllm_answer=vllm_answer,
                hf_raw=hf_raw,
                vllm_raw=vllm_raw,
                hf_score=hf_score,
                vllm_score=vllm_score,
                index=index,
            )
        )

    _write_jsonl(output, rows)
    return _summary(rows, output)


def _comparison_row(
    *,
    item: EvalManifestItem,
    hf_answer: str,
    vllm_answer: str,
    hf_raw: Any,
    vllm_raw: Any,
    hf_score: ScoreBreakdown,
    vllm_score: ScoreBreakdown,
    index: int,
) -> dict[str, Any]:
    return {
        "id": item.id or index,
        "question": item.question,
        "expected_answer": item.answer,
        "hf_answer": hf_answer,
        "vllm_answer": vllm_answer,
        "normalized_hf_answer": _normalize_answer(hf_answer),
        "normalized_vllm_answer": _normalize_answer(vllm_answer),
        "hf_pass": hf_score.passed,
        "vllm_pass": vllm_score.passed,
        "hf_points": hf_score.points,
        "vllm_points": vllm_score.points,
        "exact_match": _normalize_answer(hf_answer) == _normalize_answer(vllm_answer),
        "raw_hf_response": hf_raw,
        "raw_vllm_response": vllm_raw,
    }


def _normalize_answer(answer: str) -> str:
    return " ".join(answer.casefold().strip().split())


def _write_jsonl(output: Path, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _summary(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    return {
        "total": len(rows),
        "exact_matches": sum(1 for row in rows if row["exact_match"]),
        "hf_passes": sum(1 for row in rows if row["hf_pass"] is True),
        "vllm_passes": sum(1 for row in rows if row["vllm_pass"] is True),
        "both_pass": sum(1 for row in rows if row["hf_pass"] is True and row["vllm_pass"] is True),
        "output": str(output),
    }


def _gateway_generate_endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.path.endswith(("/generate", "/v1/chat/completions")):
        return endpoint
    return f"{endpoint}/generate"
