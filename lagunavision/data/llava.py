from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

ImageMode = Literal["reference", "copy", "symlink"]


@dataclass(frozen=True)
class LlavaExample:
    id: str
    image: Any
    question: str
    answer: str


@dataclass(frozen=True)
class LlavaMaterializeResult:
    train_manifest: Path
    eval_manifest: Path | None
    train_count: int
    eval_count: int


def materialize_llava_json(
    source_json: Path,
    output_dir: Path,
    *,
    image_roots: Iterable[Path] = (),
    limit: int = 0,
    eval_count: int = 0,
    image_mode: ImageMode = "reference",
) -> LlavaMaterializeResult:
    """Convert local LLaVA JSON/JSONL data into Laguna Vision manifests."""

    source_json = source_json.expanduser().resolve()
    image_roots = tuple(root.expanduser().resolve() for root in image_roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = output_dir / "train.jsonl"
    eval_manifest = output_dir / "eval.jsonl" if eval_count > 0 else None

    train_count = 0
    heldout_count = 0
    written = 0
    with train_manifest.open("w", encoding="utf-8") as train_handle:
        eval_handle = eval_manifest.open("w", encoding="utf-8") if eval_manifest else None
        try:
            for row in _iter_records(source_json):
                for example in _examples_from_row(row):
                    if limit > 0 and written >= limit:
                        return LlavaMaterializeResult(train_manifest, eval_manifest, train_count, heldout_count)
                    split = "eval" if heldout_count < eval_count else "train"
                    image = _materialize_image(
                        example.image,
                        source_json=source_json,
                        image_roots=image_roots,
                        output_dir=output_dir,
                        split=split,
                        index=written,
                        image_mode=image_mode,
                    )
                    handle = eval_handle if split == "eval" else train_handle
                    assert handle is not None
                    handle.write(json.dumps(_manifest_row(example, image), ensure_ascii=False) + "\n")
                    if split == "eval":
                        heldout_count += 1
                    else:
                        train_count += 1
                    written += 1
        finally:
            if eval_handle is not None:
                eval_handle.close()
    return LlavaMaterializeResult(train_manifest, eval_manifest, train_count, heldout_count)


def materialize_llava_hf(
    dataset: str,
    output_dir: Path,
    *,
    split: str = "train",
    limit: int = 0,
    eval_count: int = 0,
) -> LlavaMaterializeResult:
    """Stream a Hugging Face image/conversation dataset into Laguna Vision manifests."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install data dependencies with `python -m pip install -e '.[data]'`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = output_dir / "train.jsonl"
    eval_manifest = output_dir / "eval.jsonl" if eval_count > 0 else None
    train_count = 0
    heldout_count = 0
    written = 0

    rows = load_dataset(dataset, split=split, streaming=True)
    with train_manifest.open("w", encoding="utf-8") as train_handle:
        eval_handle = eval_manifest.open("w", encoding="utf-8") if eval_manifest else None
        try:
            for row in rows:
                for example in _examples_from_row(row):
                    if limit > 0 and written >= limit:
                        return LlavaMaterializeResult(train_manifest, eval_manifest, train_count, heldout_count)
                    target_split = "eval" if heldout_count < eval_count else "train"
                    image = _materialize_hf_image(example.image, output_dir=output_dir, split=target_split, index=written)
                    handle = eval_handle if target_split == "eval" else train_handle
                    assert handle is not None
                    handle.write(json.dumps(_manifest_row(example, image), ensure_ascii=False) + "\n")
                    if target_split == "eval":
                        heldout_count += 1
                    else:
                        train_count += 1
                    written += 1
        finally:
            if eval_handle is not None:
                eval_handle.close()
    return LlavaMaterializeResult(train_manifest, eval_manifest, train_count, heldout_count)


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"{path} must contain a JSON list or JSONL records")
            yield from data
            return
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def _examples_from_row(row: dict[str, Any]) -> list[LlavaExample]:
    image = row.get("image") or row.get("image_path") or row.get("path")
    if image is None:
        return []
    row_id = str(row.get("id") or row.get("uid") or row.get("sample_id") or _safe_name(str(image)))
    conversations = row.get("conversations") or row.get("conversation") or row.get("messages")
    if isinstance(conversations, list) and conversations:
        return _conversation_examples(row_id, image, conversations)

    answer = row.get("caption") or row.get("text") or row.get("answer") or row.get("response")
    if not answer:
        return []
    question = row.get("question") or row.get("prompt") or "Describe this image in detail."
    return [LlavaExample(id=row_id, image=image, question=_clean_prompt(str(question)), answer=str(answer).strip())]


def _conversation_examples(row_id: str, image: Any, conversations: list[dict[str, Any]]) -> list[LlavaExample]:
    examples: list[LlavaExample] = []
    history: list[tuple[str, str]] = []
    turn_index = 0
    for turn in conversations:
        role = str(turn.get("from") or turn.get("role") or "").lower()
        value = _clean_prompt(str(turn.get("value") or turn.get("content") or ""))
        if not value:
            continue
        if role in {"human", "user"}:
            history.append(("Human", value))
            continue
        if role not in {"gpt", "assistant", "model"} or not history:
            continue
        answer = value.strip()
        question = "\n".join(f"{speaker}: {text}" for speaker, text in history)
        examples.append(LlavaExample(id=f"{row_id}_{turn_index}", image=image, question=question, answer=answer))
        history.append(("Assistant", answer))
        turn_index += 1
    return examples


def _manifest_row(example: LlavaExample, image: Path) -> dict[str, Any]:
    terms = _answer_terms(example.answer)
    return {
        "id": example.id,
        "image": str(image),
        "question": example.question,
        "ocr_text": "",
        "must_include": terms,
        "accepted_fix_terms": terms,
        "must_not_include": [],
        "rubric": "instruction",
        "answer": example.answer,
    }


def _clean_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("<image>", " ")).strip()


def _answer_terms(answer: str, limit: int = 6) -> list[str]:
    stop = {"the", "and", "with", "that", "this", "from", "there", "their", "into", "about", "image"}
    words = [word for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", answer.lower()) if word not in stop]
    if words:
        return words[:limit]
    answer = answer.strip()
    return [answer[:80]] if answer else []


def _materialize_image(
    image: Any,
    *,
    source_json: Path,
    image_roots: tuple[Path, ...],
    output_dir: Path,
    split: str,
    index: int,
    image_mode: ImageMode,
) -> Path:
    source = _resolve_image_path(str(image), source_json, image_roots)
    if image_mode == "reference":
        return source
    target = _target_image_path(source.name, output_dir, split, index)
    if target.exists():
        return target
    if image_mode == "copy":
        shutil.copy2(source, target)
    elif image_mode == "symlink":
        target.symlink_to(source)
    else:
        raise ValueError(f"unsupported image mode: {image_mode}")
    return target


def _materialize_hf_image(image: Any, *, output_dir: Path, split: str, index: int) -> Path:
    if hasattr(image, "save"):
        target = _target_image_path("image.png", output_dir, split, index)
        image.save(target)
        return target
    path = Path(str(image)).expanduser()
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(f"Hugging Face row image is not a saveable image or existing path: {image!r}")


def _resolve_image_path(image: str, source_json: Path, image_roots: tuple[Path, ...]) -> Path:
    path = Path(image).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(source_json.parent / path)
        for root in image_roots:
            candidates.extend((root / path, root / "images" / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates[:8])
    raise FileNotFoundError(f"could not resolve image {image!r}; searched {searched}")


def _target_image_path(name: str, output_dir: Path, split: str, index: int) -> Path:
    suffix = Path(name).suffix or ".png"
    target_dir = output_dir / split / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{index:08d}_{_safe_name(Path(name).stem)}{suffix}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "image"
