from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping

from lagunavision.data.llava import _answer_terms, _clean_prompt, _examples_from_row, _iter_records, _safe_name
from lagunavision.data.manifest import load_manifest
from lagunavision.data.spatial_ocr import generate_spatial_ocr_manifest

RecipeStage = Literal["alignment", "instruction"]
RecipeSourceKind = Literal[
    "llava_hub_json",
    "hf_vqa",
    "hf_websight",
    "hf_rico_screenqa",
    "hf_rico_screen2words",
    "hf_websrc",
    "hf_gqa",
    "synthetic_spatial_ocr",
]

PILOT_300K_RECIPE = "general-vision-300k-v1"
COCO_TRAIN2017_URL = "http://images.cocodataset.org/zips/train2017.zip"
HF_LOAD_RETRY_DELAYS_SECONDS = (15, 45, 90)


@dataclass(frozen=True)
class RecipeSource:
    id: str
    stage: RecipeStage
    kind: RecipeSourceKind
    train_count: int
    eval_count: int
    purpose: str
    dataset: str = ""
    split: str = "train"
    config: str = ""
    source_file: str = ""
    image_archive: str = ""
    answer_style: str = "instruction"

    @property
    def total_count(self) -> int:
        return self.train_count + self.eval_count


@dataclass(frozen=True)
class RawExample:
    id: str
    image: Any
    question: str
    answer: str
    ocr_text: str = ""
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GeneralRecipeResult:
    output_dir: Path
    recipe_file: Path
    alignment_train_manifest: Path
    alignment_eval_manifest: Path
    instruction_train_manifest: Path
    instruction_eval_manifest: Path
    wrong_manifest: Path
    blank_manifest: Path
    counts: Mapping[str, int]


GENERAL_VISION_300K_SOURCES: tuple[RecipeSource, ...] = (
    RecipeSource(
        id="llava-pretrain-558k",
        stage="alignment",
        kind="llava_hub_json",
        dataset="liuhaotian/LLaVA-Pretrain",
        source_file="blip_laion_cc_sbu_558k.json",
        image_archive="images.zip",
        train_count=120_000,
        eval_count=1_000,
        answer_style="description",
        purpose="broad LLaVA-style image/caption alignment from BLIP captions over LAION/CC/SBU images",
    ),
    RecipeSource(
        id="llava-instruct-150k",
        stage="instruction",
        kind="llava_hub_json",
        dataset="liuhaotian/LLaVA-Instruct-150K",
        source_file="llava_instruct_150k.json",
        train_count=65_000,
        eval_count=700,
        purpose="general image conversation, detailed description, reasoning, and spatial questions on COCO train2017",
    ),
    RecipeSource(
        id="sharegpt4v-100k",
        stage="instruction",
        kind="llava_hub_json",
        dataset="Lin-Chen/ShareGPT4V",
        source_file="sharegpt4v_instruct_gpt4-vision_cap100k.json",
        train_count=35_000,
        eval_count=350,
        answer_style="description",
        purpose="dense GPT-4V-style visual descriptions for stronger open-ended perception on COCO train2017",
    ),
    RecipeSource(
        id="gqa-balanced",
        stage="instruction",
        kind="hf_gqa",
        dataset="lmms-lab/GQA",
        config="train_balanced_instructions",
        split="train",
        train_count=25_000,
        eval_count=500,
        purpose="object, attribute, compositional, and positional visual QA",
    ),
    RecipeSource(
        id="documentvqa",
        stage="instruction",
        kind="hf_vqa",
        dataset="HuggingFaceM4/DocumentVQA",
        split="train",
        train_count=7_000,
        eval_count=200,
        purpose="documents, forms, page layout, and OCR-heavy reading",
    ),
    RecipeSource(
        id="textvqa",
        stage="instruction",
        kind="hf_vqa",
        dataset="lmms-lab/textvqa",
        split="train",
        train_count=7_000,
        eval_count=200,
        purpose="natural images with embedded scene text",
    ),
    RecipeSource(
        id="ocr-vqa",
        stage="instruction",
        kind="hf_vqa",
        dataset="howard-hou/OCR-VQA",
        split="train",
        train_count=6_000,
        eval_count=200,
        purpose="book-cover OCR and text-grounded visual QA",
    ),
    RecipeSource(
        id="chartqa",
        stage="instruction",
        kind="hf_vqa",
        dataset="HuggingFaceM4/ChartQA",
        split="train",
        train_count=5_000,
        eval_count=150,
        purpose="chart, plot, and quantitative visual QA",
    ),
    RecipeSource(
        id="websight",
        stage="instruction",
        kind="hf_websight",
        dataset="HuggingFaceM4/WebSight",
        split="train",
        train_count=10_000,
        eval_count=250,
        answer_style="description",
        purpose="synthetic webpage screenshots with HTML-derived layout and purpose descriptions",
    ),
    RecipeSource(
        id="rico-screenqa",
        stage="instruction",
        kind="hf_rico_screenqa",
        dataset="rootsautomation/RICO-ScreenQA",
        split="train",
        train_count=7_000,
        eval_count=200,
        purpose="mobile app screenshot QA and UI element understanding",
    ),
    RecipeSource(
        id="rico-screen2words",
        stage="instruction",
        kind="hf_rico_screen2words",
        dataset="rootsautomation/RICO-Screen2Words",
        split="train",
        train_count=4_000,
        eval_count=150,
        answer_style="description",
        purpose="mobile screenshot captioning and screen-level semantics",
    ),
    RecipeSource(
        id="websrc",
        stage="instruction",
        kind="hf_websrc",
        dataset="rootsautomation/websrc",
        split="train",
        train_count=4_000,
        eval_count=150,
        purpose="web screenshot QA grounded in page text and layout",
    ),
    RecipeSource(
        id="spatial-ocr",
        stage="instruction",
        kind="synthetic_spatial_ocr",
        train_count=5_000,
        eval_count=500,
        purpose="hard positional OCR controls for top-left/top-right/bottom-left/bottom-right grounding",
    ),
)


def get_recipe_sources(
    recipe: str = PILOT_300K_RECIPE,
    *,
    sample_per_source: int = 0,
    train_budget: int = 0,
) -> tuple[RecipeSource, ...]:
    if recipe != PILOT_300K_RECIPE:
        raise ValueError(f"unsupported recipe {recipe!r}; expected {PILOT_300K_RECIPE!r}")
    if sample_per_source > 0 and train_budget > 0:
        raise ValueError("use either sample_per_source or train_budget, not both")
    if train_budget > 0:
        return _scale_sources_to_train_budget(GENERAL_VISION_300K_SOURCES, train_budget)
    if sample_per_source <= 0:
        return GENERAL_VISION_300K_SOURCES
    adjusted: list[RecipeSource] = []
    for source in GENERAL_VISION_300K_SOURCES:
        train_count = min(source.train_count, sample_per_source)
        eval_count = min(source.eval_count, max(1, sample_per_source // 5))
        adjusted.append(replace(source, train_count=train_count, eval_count=eval_count))
    return tuple(adjusted)


def recipe_summary(
    recipe: str = PILOT_300K_RECIPE,
    *,
    sample_per_source: int = 0,
    train_budget: int = 0,
) -> dict[str, Any]:
    sources = get_recipe_sources(recipe, sample_per_source=sample_per_source, train_budget=train_budget)
    by_stage = Counter()
    for source in sources:
        by_stage[f"{source.stage}_train"] += source.train_count
        by_stage[f"{source.stage}_eval"] += source.eval_count
    return {
        "recipe": recipe,
        "total_train": by_stage["alignment_train"] + by_stage["instruction_train"],
        "total_eval": by_stage["alignment_eval"] + by_stage["instruction_eval"],
        "alignment_train": by_stage["alignment_train"],
        "alignment_eval": by_stage["alignment_eval"],
        "instruction_train": by_stage["instruction_train"],
        "instruction_eval": by_stage["instruction_eval"],
        "sources": [asdict(source) for source in sources],
    }


def _scale_sources_to_train_budget(sources: tuple[RecipeSource, ...], train_budget: int) -> tuple[RecipeSource, ...]:
    total_train = sum(source.train_count for source in sources)
    if train_budget <= 0:
        return sources
    if train_budget >= total_train:
        return sources

    scale = train_budget / total_train
    floors: list[int] = []
    remainders: list[tuple[float, int]] = []
    for index, source in enumerate(sources):
        raw = source.train_count * scale
        count = max(1, int(raw))
        floors.append(count)
        remainders.append((raw - int(raw), index))

    remaining = train_budget - sum(floors)
    for _, index in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        floors[index] += 1

    adjusted: list[RecipeSource] = []
    for source, train_count in zip(sources, floors):
        eval_count = max(1, round(source.eval_count * scale))
        adjusted.append(replace(source, train_count=train_count, eval_count=eval_count))
    return tuple(adjusted)


def materialize_general_recipe(
    output_dir: Path,
    *,
    recipe: str = PILOT_300K_RECIPE,
    sample_per_source: int = 0,
    train_budget: int = 0,
    download_assets: bool = False,
    coco_train2017_root: Path | None = None,
    llava_pretrain_image_root: Path | None = None,
    seed: int = 7,
) -> GeneralRecipeResult:
    output_dir = output_dir.expanduser().resolve()
    sources = get_recipe_sources(recipe, sample_per_source=sample_per_source, train_budget=train_budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment").mkdir(exist_ok=True)
    (output_dir / "instruction").mkdir(exist_ok=True)
    (output_dir / "controls").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)

    recipe_file = output_dir / "recipe.json"
    recipe_file.write_text(
        json.dumps(recipe_summary(recipe, sample_per_source=sample_per_source, train_budget=train_budget), indent=2),
        encoding="utf-8",
    )

    manifests = {
        ("alignment", "train"): output_dir / "alignment" / "train.jsonl",
        ("alignment", "eval"): output_dir / "alignment" / "eval.jsonl",
        ("instruction", "train"): output_dir / "instruction" / "train.jsonl",
        ("instruction", "eval"): output_dir / "instruction" / "eval.jsonl",
    }
    handles = {key: path.open("w", encoding="utf-8") for key, path in manifests.items()}
    counts: Counter[str] = Counter()
    try:
        for source in sources:
            total = 0
            for example in _iter_source_examples(
                source,
                output_dir=output_dir,
                download_assets=download_assets,
                coco_train2017_root=coco_train2017_root,
                llava_pretrain_image_root=llava_pretrain_image_root,
                seed=seed,
            ):
                if total >= source.total_count:
                    break
                split = "eval" if total < source.eval_count else "train"
                manifest = manifests[(source.stage, split)]
                image = _materialize_example_image(example.image, output_dir, source, split, counts[f"{source.id}_images"])
                row = _manifest_row(source, example, image, manifest)
                handles[(source.stage, split)].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"{source.stage}_{split}"] += 1
                counts[f"{source.id}_{split}"] += 1
                counts[f"{source.id}_images"] += 1
                total += 1
            if total < source.total_count:
                raise RuntimeError(f"{source.id} produced {total} examples, expected {source.total_count}")
    finally:
        for handle in handles.values():
            handle.close()

    for path in manifests.values():
        validate_manifest_images(path)

    wrong_manifest, blank_manifest = write_control_manifests(
        manifests[("instruction", "eval")],
        output_dir / "controls",
    )
    validate_manifest_images(wrong_manifest)
    validate_manifest_images(blank_manifest)

    return GeneralRecipeResult(
        output_dir=output_dir,
        recipe_file=recipe_file,
        alignment_train_manifest=manifests[("alignment", "train")],
        alignment_eval_manifest=manifests[("alignment", "eval")],
        instruction_train_manifest=manifests[("instruction", "train")],
        instruction_eval_manifest=manifests[("instruction", "eval")],
        wrong_manifest=wrong_manifest,
        blank_manifest=blank_manifest,
        counts=dict(counts),
    )


def write_control_manifests(eval_manifest: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_manifest(eval_manifest)
    if len(items) < 2:
        raise ValueError("at least two eval examples are required to build wrong-image controls")

    blank_image = output_dir / "blank.png"
    if not blank_image.exists():
        from PIL import Image

        Image.new("RGB", (1024, 1024), "white").save(blank_image)

    wrong_manifest = output_dir / "wrong.jsonl"
    blank_manifest = output_dir / "blank.jsonl"
    with wrong_manifest.open("w", encoding="utf-8") as wrong_handle, blank_manifest.open("w", encoding="utf-8") as blank_handle:
        for index, item in enumerate(items):
            wrong_image = items[(index + 1) % len(items)].image
            base = {
                "id": item.id,
                "question": item.question,
                "ocr_text": item.ocr_text,
                "must_include": list(item.must_include),
                "accepted_fix_terms": list(item.accepted_fix_terms),
                "must_not_include": list(item.must_not_include),
                "answer": item.answer,
            }
            wrong_handle.write(
                json.dumps(
                    {
                        **base,
                        "id": f"{item.id}__wrong_image",
                        "image": _image_value_for_manifest(wrong_image, wrong_manifest),
                        "rubric": "wrong-image-control",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            blank_handle.write(
                json.dumps(
                    {
                        **base,
                        "id": f"{item.id}__blank_image",
                        "image": _image_value_for_manifest(blank_image, blank_manifest),
                        "rubric": "blank-image-control",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return wrong_manifest, blank_manifest


def validate_manifest_images(manifest: Path) -> int:
    rows = 0
    required = {"id", "image", "question", "answer"}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{manifest}:{line_number} missing fields: {', '.join(missing)}")
            image = Path(str(row["image"]))
            if not image.is_absolute():
                image = manifest.parent / image
            if not image.exists():
                raise FileNotFoundError(f"{manifest}:{line_number} image does not exist: {image}")
            rows += 1
    if rows == 0:
        raise ValueError(f"{manifest} is empty")
    return rows


def _iter_source_examples(
    source: RecipeSource,
    *,
    output_dir: Path,
    download_assets: bool,
    coco_train2017_root: Path | None,
    llava_pretrain_image_root: Path | None,
    seed: int,
) -> Iterator[RawExample]:
    if source.kind == "llava_hub_json":
        yield from _iter_llava_hub_json(
            source,
            output_dir=output_dir,
            download_assets=download_assets,
            coco_train2017_root=coco_train2017_root,
            llava_pretrain_image_root=llava_pretrain_image_root,
        )
    elif source.kind == "hf_gqa":
        yield from _iter_gqa(source, output_dir=output_dir)
    elif source.kind == "synthetic_spatial_ocr":
        yield from _iter_spatial_ocr(source, output_dir=output_dir, seed=seed)
    else:
        yield from _iter_hf_rows(source)


def _iter_llava_hub_json(
    source: RecipeSource,
    *,
    output_dir: Path,
    download_assets: bool,
    coco_train2017_root: Path | None,
    llava_pretrain_image_root: Path | None,
) -> Iterator[RawExample]:
    hub_download = _require_huggingface_hub()
    source_json = Path(hub_download(repo_id=source.dataset, filename=source.source_file, repo_type="dataset"))
    image_roots: list[Path] = []
    if source.id == "llava-pretrain-558k":
        extract_dir = output_dir / "raw" / "llava_pretrain"
        if llava_pretrain_image_root is not None:
            image_roots.append(llava_pretrain_image_root)
            extract_dir = llava_pretrain_image_root.parent if llava_pretrain_image_root.name == "images" else llava_pretrain_image_root
        image_roots.extend((output_dir / "raw" / "llava_pretrain", output_dir / "raw" / "llava_pretrain" / "images"))
        if download_assets and not any(root.exists() for root in image_roots):
            archive = Path(hub_download(repo_id=source.dataset, filename=source.image_archive, repo_type="dataset"))
            _extract_zip_once(archive, extract_dir)
    else:
        coco_root = _ensure_coco_root(output_dir, download_assets=download_assets, coco_train2017_root=coco_train2017_root)
        image_roots.extend((coco_root, coco_root.parent, output_dir / "raw" / "coco"))

    for row in _iter_records(source_json):
        for llava_example in _examples_from_row(row):
            image = _resolve_image_path(str(llava_example.image), source_json, image_roots)
            yield RawExample(
                id=f"{source.id}:{llava_example.id}",
                image=image,
                question=llava_example.question,
                answer=llava_example.answer,
            )


def _iter_gqa(source: RecipeSource, *, output_dir: Path) -> Iterator[RawExample]:
    load_dataset = _require_datasets()
    target = source.total_count
    question_rows: list[Mapping[str, Any]] = []
    needed_images: set[str] = set()
    rows = _load_hf_dataset_with_retries(
        load_dataset,
        source.dataset,
        name="train_balanced_instructions",
        split=source.split,
        streaming=True,
    )
    for row in rows:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("fullAnswer") or row.get("answer") or "").strip()
        image_id = str(row.get("imageId") or "").strip()
        if not question or not answer or not image_id:
            continue
        question_rows.append(row)
        needed_images.add(image_id)
        if len(question_rows) >= target:
            break

    image_paths: dict[str, Path] = {}
    image_rows = _load_hf_dataset_with_retries(
        load_dataset,
        source.dataset,
        name="train_balanced_images",
        split=source.split,
        streaming=True,
    )
    image_dir = output_dir / "images" / source.id
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in image_rows:
        image_id = str(row.get("id") or "").strip()
        if image_id not in needed_images:
            continue
        image_paths[image_id] = _save_pil_image(row["image"], image_dir / f"{_safe_name(image_id)}.jpg")
        if len(image_paths) >= len(needed_images):
            break

    for index, row in enumerate(question_rows):
        image_id = str(row["imageId"])
        if image_id not in image_paths:
            continue
        yield RawExample(
            id=f"{source.id}:{row.get('id') or index}",
            image=image_paths[image_id],
            question=str(row["question"]).strip(),
            answer=str(row.get("fullAnswer") or row.get("answer")).strip(),
            metadata={"image_id": image_id, "semantic": row.get("semanticStr", "")},
        )


def _iter_hf_rows(source: RecipeSource) -> Iterator[RawExample]:
    load_dataset = _require_datasets()
    kwargs: dict[str, Any] = {"split": source.split, "streaming": True}
    if source.config:
        kwargs["name"] = source.config
    rows = _load_hf_dataset_with_retries(load_dataset, source.dataset, **kwargs)
    for index, row in enumerate(rows):
        if source.id == "ocr-vqa":
            yield from _examples_from_ocr_vqa_row(source, row, index)
            continue
        example = _example_from_hf_row(source, row, index)
        if example is not None:
            yield example


def _load_hf_dataset_with_retries(load_dataset: Any, dataset: str, **kwargs: Any) -> Any:
    for attempt in range(len(HF_LOAD_RETRY_DELAYS_SECONDS) + 1):
        try:
            return load_dataset(dataset, **kwargs)
        except Exception as exc:
            if attempt >= len(HF_LOAD_RETRY_DELAYS_SECONDS) or not _is_transient_hf_load_error(exc):
                raise
            delay = HF_LOAD_RETRY_DELAYS_SECONDS[attempt]
            print(
                f"hf_load_dataset_retry dataset={dataset} attempt={attempt + 1} delay_seconds={delay} error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"failed to load dataset after retries: {dataset}")


def _is_transient_hf_load_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "504 gateway",
            "gateway time-out",
            "gateway timeout",
            "temporarily unavailable",
            "too many requests",
            "connection reset",
            "read timed out",
            "timeout",
        )
    )


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _examples_from_ocr_vqa_row(source: RecipeSource, row: Mapping[str, Any], index: int) -> Iterator[RawExample]:
    image = _first_present(row, "image", "img", "image_id", "image_path")
    if image is None:
        return

    questions = _as_string_list(_first_present(row, "questions", "question"))
    answers = _as_string_list(_first_present(row, "answers", "answer"))
    if len(answers) == 1 and len(questions) > 1:
        answers = answers * len(questions)

    ocr_tokens = _as_string_list(row.get("ocr_tokens"))
    metadata = {
        key: row[key]
        for key in ("image_id", "title", "authorName", "genre", "set_name")
        if row.get(key) not in (None, "")
    }
    base_id = _row_id(row, index)
    for qa_index, (question, answer) in enumerate(zip(questions, answers)):
        if not question or not answer:
            continue
        yield RawExample(
            id=f"{source.id}:{base_id}:{qa_index}",
            image=image,
            question=question,
            answer=answer,
            ocr_text=" ".join(ocr_tokens),
            metadata=metadata,
        )


def _example_from_hf_row(source: RecipeSource, row: Mapping[str, Any], index: int) -> RawExample | None:
    if source.kind == "hf_vqa":
        question = _first_string(row, "question", "query", "prompt")
        answer = _answer_from_row(row)
        image = _first_present(row, "image", "img", "image_id", "image_path")
        if not question or not answer or image is None:
            return None
        return RawExample(id=f"{source.id}:{_row_id(row, index)}", image=image, question=question, answer=answer)

    if source.kind == "hf_websight":
        image = row.get("image")
        idea = _first_string(row, "llm_generated_idea", "prompt")
        html_text = _html_to_text(_first_string(row, "text", "html"))
        answer = idea or html_text[:600]
        if image is None or not answer:
            return None
        question = "Describe this webpage screenshot, including its layout, visible UI elements, text, and likely purpose."
        if html_text:
            answer = f"{answer}\n\nVisible/page text cues: {html_text[:400]}".strip()
        return RawExample(id=f"{source.id}:{_row_id(row, index)}", image=image, question=question, answer=answer)

    if source.kind == "hf_rico_screenqa":
        image = row.get("image")
        question = _first_string(row, "question")
        answer = _rico_ground_truth(row)
        if image is None or not question or not answer:
            return None
        return RawExample(id=f"{source.id}:{_row_id(row, index)}", image=image, question=question, answer=answer)

    if source.kind == "hf_rico_screen2words":
        image = row.get("image")
        captions = row.get("captions") or []
        if isinstance(captions, list):
            answer = max((str(caption).strip() for caption in captions), key=len, default="")
        else:
            answer = str(captions).strip()
        if image is None or not answer:
            return None
        return RawExample(
            id=f"{source.id}:{_row_id(row, index)}",
            image=image,
            question="Describe this mobile app screenshot, including its main purpose and visible UI elements.",
            answer=answer,
        )

    if source.kind == "hf_websrc":
        image = row.get("image")
        question = _first_string(row, "question")
        answer = _answer_from_row(row)
        if image is None or not question or not answer:
            return None
        return RawExample(id=f"{source.id}:{_row_id(row, index)}", image=image, question=question, answer=answer)

    raise ValueError(f"unsupported HF source kind: {source.kind}")


def _iter_spatial_ocr(source: RecipeSource, *, output_dir: Path, seed: int) -> Iterator[RawExample]:
    source_dir = output_dir / "raw" / "spatial_ocr"
    examples = generate_spatial_ocr_manifest(source_dir, source.total_count, seed=seed)
    for example in examples:
        yield RawExample(
            id=f"{source.id}:{example.id}",
            image=source_dir / example.image,
            question=example.question,
            answer=example.answer,
        )


def _manifest_row(source: RecipeSource, example: RawExample, image: Path, manifest: Path) -> dict[str, Any]:
    terms = _answer_terms(example.answer)
    row = {
        "id": example.id,
        "image": _image_value_for_manifest(image, manifest),
        "question": _clean_prompt(example.question),
        "ocr_text": example.ocr_text,
        "must_include": terms,
        "accepted_fix_terms": terms,
        "must_not_include": [],
        "rubric": source.answer_style,
        "answer": example.answer.strip(),
        "source_id": source.id,
        "source_dataset": source.dataset or "lagunavision/synthetic",
        "source_stage": source.stage,
    }
    if example.metadata:
        row["source_metadata"] = dict(example.metadata)
    return row


def _materialize_example_image(image: Any, output_dir: Path, source: RecipeSource, split: str, index: int) -> Path:
    if isinstance(image, Path):
        if image.exists():
            return image.resolve()
        raise FileNotFoundError(f"image path does not exist: {image}")
    if hasattr(image, "save"):
        return _save_pil_image(image, output_dir / "images" / source.id / split / f"{index:08d}.jpg")
    if isinstance(image, str):
        path = Path(image).expanduser()
        path_exists = False
        try:
            path_exists = path.exists()
        except OSError:
            # Long inline base64 images are not filesystem paths.
            pass
        if path_exists:
            return path.resolve()
        if image.startswith("data:image/"):
            image = image.split(",", 1)[1]
        if _looks_like_base64_image(image):
            return _save_base64_image(image, output_dir / "images" / source.id / split / f"{index:08d}.jpg")
        return _save_base64_image(image, output_dir / "images" / source.id / split / f"{index:08d}.jpg")
    raise TypeError(f"unsupported image value for {source.id}: {type(image).__name__}")


def _looks_like_base64_image(value: str) -> bool:
    text = value.strip()
    return len(text) > 256 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", text) is not None


def _save_pil_image(image: Any, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target.resolve()
    image = image.convert("RGB") if hasattr(image, "convert") else image
    image.save(target, quality=95)
    return target.resolve()


def _save_base64_image(value: str, target: Path) -> Path:
    try:
        image_bytes = base64.b64decode(value, validate=True)
    except Exception as exc:  # noqa: BLE001 - normalize all decode failures into one useful error.
        raise FileNotFoundError(f"image string is neither an existing path nor valid base64 data: {value[:40]!r}") from exc
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    return _save_pil_image(image, target)


def _image_value_for_manifest(image: Path, manifest: Path) -> str:
    image = image.expanduser().resolve()
    try:
        return os.path.relpath(image, manifest.parent)
    except ValueError:
        return str(image)


def _resolve_image_path(image: str, source_json: Path, image_roots: Iterable[Path]) -> Path:
    path = Path(image).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(source_json.parent / path)
        name = path.name
        for root in image_roots:
            root = root.expanduser()
            root_variants = [root]
            if root.name == "images":
                root_variants.append(root.parent)
            else:
                root_variants.append(root / "images")
            root_variants.append(root.parent / "images")
            seen_roots: set[Path] = set()
            for root_variant in root_variants:
                if root_variant in seen_roots:
                    continue
                seen_roots.add(root_variant)
                candidates.extend(
                    (
                        root_variant / path,
                        root_variant / name,
                        root_variant / "train2017" / name,
                        root_variant / "coco" / "train2017" / name,
                    )
                )
            candidates.extend(
                (
                    root / path,
                    root / "images" / path,
                    root / name,
                )
            )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates[:10])
    raise FileNotFoundError(f"could not resolve image {image!r}; searched {searched}")


def _ensure_coco_root(output_dir: Path, *, download_assets: bool, coco_train2017_root: Path | None) -> Path:
    candidates = []
    if coco_train2017_root is not None:
        candidates.append(coco_train2017_root.expanduser())
    candidates.extend((output_dir / "raw" / "coco" / "train2017", output_dir / "raw" / "coco"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if not download_assets:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "COCO train2017 images are required for LLaVA-Instruct/ShareGPT4V. "
            f"Set --coco-train2017-root or pass --download-assets. Searched: {searched}"
        )
    if coco_train2017_root is not None:
        root = coco_train2017_root.expanduser().parent if coco_train2017_root.name == "train2017" else coco_train2017_root.expanduser()
    else:
        root = output_dir / "raw" / "coco"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "train2017.zip"
    if not archive.exists():
        urllib.request.urlretrieve(COCO_TRAIN2017_URL, archive)
    _extract_zip_once(archive, root)
    return (root / "train2017").resolve()


def _extract_zip_once(archive: Path, target_dir: Path) -> None:
    marker = target_dir / ".extracted"
    if marker.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_handle:
        zip_handle.extractall(target_dir)
    marker.write_text(str(archive), encoding="utf-8")


def _require_huggingface_hub() -> Any:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install data dependencies with `python -m pip install -e '.[data]'`.") from exc
    return hf_hub_download


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install data dependencies with `python -m pip install -e '.[data]'`.") from exc
    return load_dataset


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _first_string(row: Mapping[str, Any], *keys: str) -> str:
    value = _first_present(row, *keys)
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def _answer_from_row(row: Mapping[str, Any]) -> str:
    value = _first_present(row, "answer", "answers", "label", "response")
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        counts = Counter(str(item).strip() for item in value if str(item).strip())
        if counts:
            return counts.most_common(1)[0][0]
        return ""
    if isinstance(value, dict):
        for key in ("answer", "text", "label"):
            if value.get(key):
                return str(value[key]).strip()
        return json.dumps(value, sort_keys=True)
    return str(value).strip()


def _rico_ground_truth(row: Mapping[str, Any]) -> str:
    ground_truth = row.get("ground_truth") or row.get("answers")
    if isinstance(ground_truth, list):
        values = []
        for item in ground_truth:
            if isinstance(item, Mapping):
                values.extend(str(item.get(key, "")).strip() for key in ("full_answer", "answer", "text"))
            else:
                values.append(str(item).strip())
        values = [value for value in values if value]
        return max(values, key=len, default="")
    return str(ground_truth or "").strip()


def _row_id(row: Mapping[str, Any], index: int) -> str:
    return _safe_name(str(row.get("id") or row.get("question_id") or row.get("image_id") or row.get("screen_id") or index))


_TAG_RE = re.compile(r"<[^>]+>")


def _html_to_text(value: str) -> str:
    text = _TAG_RE.sub(" ", value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
