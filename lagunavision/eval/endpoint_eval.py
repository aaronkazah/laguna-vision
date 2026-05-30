from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lagunavision.data.manifest import load_manifest
from lagunavision.eval.score_eval import score_answer
from lagunavision.types import EvalManifestItem


@dataclass(frozen=True)
class EndpointEvalConfig:
    endpoint: str
    manifest: Path
    output: Path
    summary_output: Path | None = None
    token: str = ""
    max_new_tokens: int = 64
    timeout: float = 120.0
    limit: int = 0


def run_endpoint_eval(config: EndpointEvalConfig) -> dict[str, Any]:
    items = load_manifest(config.manifest)
    if config.limit > 0:
        items = items[: config.limit]
    metadata = _manifest_metadata(config.manifest)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with config.output.open("w", encoding="utf-8") as handle:
        for item in items:
            response = _ask_endpoint(config, item)
            answer = _extract_answer(response)
            score = score_answer(item, answer)
            meta = metadata.get(item.id, {})
            row = {
                "id": item.id,
                "category": meta.get("category", ""),
                "expected_result": meta.get("expected_result", ""),
                "question": item.question,
                "expected_answer": item.answer,
                "answer": answer,
                "raw_answer": _extract_raw_answer(response),
                "raw_response": response,
                "points": score.points,
                "passed": score.passed,
                "read_key_text": score.read_key_text,
                "identified_cause": score.identified_cause,
                "gave_fix": score.gave_fix,
                "violated_negative": score.violated_negative,
                "must_include": list(item.must_include),
                "must_include_mode": item.must_include_mode,
                "must_not_include": list(item.must_not_include),
                "max_answer_words": item.max_answer_words,
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = _summarize(rows)
    summary_path = config.summary_output or config.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _ask_endpoint(config: EndpointEvalConfig, item: EvalManifestItem) -> Any:
    payload = {
        "inputs": {
            "image": _image_data_uri(item.image),
            "question": item.question,
            "max_new_tokens": config.max_new_tokens,
        }
    }
    return _post_json(config.endpoint, payload, token=config.token, timeout=config.timeout)


def _image_data_uri(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _post_json(endpoint: str, payload: dict[str, Any], *, token: str, timeout: float) -> Any:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"endpoint request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"endpoint request failed: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _extract_answer(data: Any) -> str:
    if isinstance(data, str):
        return _extract_final_answer(data)
    if isinstance(data, list) and data:
        return _extract_answer(data[0])
    if isinstance(data, dict):
        answer_keys = ("content", "answer", "generated_text", "text", "message", "output", "outputs", "choices")
        if "error" in data and not any(key in data for key in answer_keys):
            raise RuntimeError(f"endpoint returned error: {data['error']}")
        for key in ("choices", "message", "output", "outputs"):
            if key in data:
                return _extract_answer(data[key])
        for key in ("content", "answer", "generated_text", "text"):
            if key in data:
                value = data[key]
                if isinstance(value, (dict, list)):
                    return _extract_answer(value)
                return _extract_final_answer(str(value))
    raise RuntimeError(f"could not extract an answer from endpoint response: {data!r}")


def _extract_raw_answer(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, list) and data:
        return _extract_raw_answer(data[0])
    if isinstance(data, dict):
        for key in ("choices", "message", "output", "outputs"):
            if key in data:
                return _extract_raw_answer(data[key])
        for key in ("content", "answer", "generated_text", "text"):
            if key in data:
                value = data[key]
                if isinstance(value, (dict, list)):
                    return _extract_raw_answer(value)
                return str(value).strip()
    return ""


def _extract_final_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    # Reasoning models commonly emit hidden/visible thought before final content
    # using <think>...</think> or a labeled final answer. Keep thinking enabled,
    # but score only the final answer segment.
    stripped = re.sub(r"(?is)<think>.*?</think>", "", stripped)
    stripped = re.sub(r"(?i)</?(?:assistant|user|system)\s*>", "\n", stripped)
    stripped = re.sub(r"(?i)<\|(?:im_start|im_end|start_header_id|end_header_id|eot_id|endoftext)\|>", "\n", stripped)
    stripped = re.sub(r"(?i)</?s>", "\n", stripped).strip()

    labeled = re.search(r"(?im)^\s*(?:final\s+answer|answer)\s*[:：]\s*(.*)$", stripped)
    if labeled:
        label_value = labeled.group(1).strip()
        if label_value:
            return _clean_answer_line(label_value)
        remaining = stripped[labeled.end() :].splitlines()
        for line in remaining:
            cleaned = _clean_answer_line(line)
            if cleaned:
                return cleaned

    lines = [_clean_answer_line(line) for line in stripped.splitlines()]
    lines = [line for line in lines if line and not _is_prompt_echo(line)]
    if len(lines) > 1:
        non_junk_lines = [line for line in lines if not _is_placeholder_junk(line)]
        if non_junk_lines:
            lines = non_junk_lines
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    short_lines = [line for line in lines if len(re.findall(r"\b[\w.$:-]+\b", line)) <= 6]
    return short_lines[-1] if short_lines else lines[-1]


def _clean_answer_line(line: str) -> str:
    cleaned = re.sub(r"(?i)</?(?:assistant|user|system)\s*>", "", line)
    cleaned = re.sub(r"(?i)<\|(?:im_start|im_end|start_header_id|end_header_id|eot_id|endoftext)\|>", "", cleaned)
    cleaned = re.sub(r"(?i)</?s>", "", cleaned)
    return cleaned.strip().strip('"').strip("'").strip()


def _is_prompt_echo(line: str) -> bool:
    normalized = line.casefold()
    return (
        normalized.startswith(("what ", "which ", "copy only ", "give the final answer"))
        or "give the final answer" in normalized
    )


def _is_placeholder_junk(line: str) -> bool:
    normalized = line.strip().casefold()
    return (
        normalized in {"<", "<no", "<no answer", "<no answer>", "no answer", "<no text", "<no text>", "no text", "unanswerable"}
        or normalized.startswith("unanswerable:")
        or normalized.startswith("<no ")
    )


def _manifest_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            metadata[str(row["id"])] = row
    return metadata


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "passed": sum(int(row["passed"]) for row in rows),
        "total": len(rows),
        "accuracy": _accuracy(rows),
        "by_category": _group_summary(rows, "category"),
        "by_expected_result": _group_summary(rows, "expected_result"),
    }
    return summary


def _group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "uncategorized")
        groups.setdefault(key, []).append(row)
    return {
        key: {
            "passed": sum(int(row["passed"]) for row in group),
            "total": len(group),
            "accuracy": _accuracy(group),
        }
        for key, group in sorted(groups.items())
    }


def _accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(int(row["passed"]) for row in rows) / len(rows)
