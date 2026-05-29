from __future__ import annotations

import gc
import json
import multiprocessing
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Iterable, Mapping


@dataclass(frozen=True)
class HfDatasetRequest:
    dataset: str
    split: str = "train"
    config: str | None = None


DEFAULT_HF_DATASETS = (
    HfDatasetRequest("HuggingFaceM4/DocumentVQA", "train"),
    HfDatasetRequest("lmms-lab/textvqa", "train"),
    HfDatasetRequest("howard-hou/OCR-VQA", "train"),
    HfDatasetRequest("HuggingFaceM4/ChartQA", "train"),
)


def materialize_hf_dataset(
    output_dir: Path,
    train_count: int,
    eval_count: int,
    datasets: tuple[HfDatasetRequest, ...] = DEFAULT_HF_DATASETS,
) -> tuple[Path, Path]:
    return _materialize_hf_dataset_isolated(output_dir, train_count, eval_count, datasets)


def _materialize_hf_dataset_streaming(
    output_dir: Path,
    train_count: int,
    eval_count: int,
    datasets: tuple[HfDatasetRequest, ...] = DEFAULT_HF_DATASETS,
) -> tuple[Path, Path]:
    if train_count <= 0 or eval_count <= 0:
        raise ValueError("train_count and eval_count must be positive")
    if not datasets:
        raise ValueError("at least one HF dataset is required")
    try:
        import datasets as hf_datasets
    except ImportError as exc:
        raise RuntimeError("Install dataset dependencies with `python -m pip install -e '.[data]'`.") from exc
    hf_datasets.disable_progress_bars()
    _disable_tqdm_monitor()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = output_dir / "train.jsonl"
    eval_manifest = output_dir / "eval.jsonl"
    train_targets = _split_count(train_count, len(datasets))
    eval_targets = _split_count(eval_count, len(datasets))
    train_written = 0
    eval_written = 0

    with train_manifest.open("w", encoding="utf-8") as train_handle, eval_manifest.open("w", encoding="utf-8") as eval_handle:
        for dataset_index, request in enumerate(datasets):
            rows = hf_datasets.load_dataset(
                request.dataset,
                name=request.config,
                split=request.split,
                streaming=True,
            )
            dataset_train_written = 0
            dataset_eval_written = 0
            dataset_train_target = train_targets[dataset_index]
            dataset_eval_target = eval_targets[dataset_index]
            for row_index, row in enumerate(rows):
                if dataset_train_written >= dataset_train_target and dataset_eval_written >= dataset_eval_target:
                    break
                normalized = _normalize_row(request.dataset, row, row_index)
                if normalized is None:
                    continue
                split = "train" if dataset_train_written < dataset_train_target else "eval"
                handle = train_handle if split == "train" else eval_handle
                item_index = train_written if split == "train" else eval_written
                image_path = _save_image(output_dir, split, item_index, normalized["image"])
                manifest_row = {
                    "id": f"{split}_{item_index:05d}_{_safe_id(request.dataset)}",
                    "image": str(image_path.relative_to(output_dir)),
                    "question": normalized["question"],
                    "ocr_text": normalized.get("ocr_text", ""),
                    "rubric": "vqa",
                    "must_include": [normalized["answer"]],
                    "accepted_fix_terms": [normalized["answer"]],
                    "must_not_include": [],
                    "source_dataset": request.dataset,
                }
                handle.write(json.dumps(manifest_row, sort_keys=True) + "\n")
                if split == "train":
                    dataset_train_written += 1
                    train_written += 1
                else:
                    dataset_eval_written += 1
                    eval_written += 1
            if dataset_train_written < dataset_train_target or dataset_eval_written < dataset_eval_target:
                raise RuntimeError(f"{request.dataset} did not provide enough usable rows")
            del rows
            gc.collect()
    if train_written != train_count or eval_written != eval_count:
        raise RuntimeError(f"wrote {train_written} train and {eval_written} eval rows, expected {train_count}/{eval_count}")
    _stop_tqdm_monitor()
    return train_manifest, eval_manifest


def _materialize_hf_dataset_isolated(
    output_dir: Path,
    train_count: int,
    eval_count: int,
    datasets: tuple[HfDatasetRequest, ...] = DEFAULT_HF_DATASETS,
) -> tuple[Path, Path]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_materialize_worker,
        args=(
            str(output_dir),
            train_count,
            eval_count,
            tuple((request.dataset, request.split, request.config) for request in datasets),
            queue,
        ),
    )
    process.start()
    message = None
    while process.is_alive():
        try:
            message = queue.get(timeout=0.5)
            break
        except Empty:
            continue
    if message is None:
        try:
            message = queue.get_nowait()
        except Empty as exc:
            process.join()
            raise RuntimeError(f"HF materializer failed with exit code {process.exitcode}") from exc

    status, payload = message
    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    if status == "error":
        raise RuntimeError(str(payload))
    train_manifest, eval_manifest = payload
    return Path(train_manifest), Path(eval_manifest)


def parse_dataset_requests(values: Iterable[str]) -> tuple[HfDatasetRequest, ...]:
    requests = []
    for value in values:
        parts = value.split(":")
        if len(parts) == 1:
            requests.append(HfDatasetRequest(parts[0]))
        elif len(parts) == 2:
            requests.append(HfDatasetRequest(parts[0], parts[1]))
        elif len(parts) == 3:
            requests.append(HfDatasetRequest(parts[0], parts[1], parts[2]))
        else:
            raise ValueError(f"invalid dataset spec: {value}")
    return tuple(requests)


def _materialize_worker(
    output_dir: str,
    train_count: int,
    eval_count: int,
    dataset_specs: tuple[tuple[str, str, str | None], ...],
    queue,
) -> None:
    try:
        train_manifest, eval_manifest = _materialize_hf_dataset_streaming(
            Path(output_dir),
            train_count,
            eval_count,
            tuple(HfDatasetRequest(*spec) for spec in dataset_specs),
        )
    except Exception as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    queue.put(("ok", (str(train_manifest), str(eval_manifest))))


def _split_count(total: int, parts: int) -> tuple[int, ...]:
    base = total // parts
    remainder = total % parts
    return tuple(base + int(index < remainder) for index in range(parts))


def _disable_tqdm_monitor() -> None:
    try:
        from tqdm.auto import tqdm
        from tqdm.std import tqdm as std_tqdm
    except ImportError:
        return
    tqdm.monitor_interval = 0
    std_tqdm.monitor_interval = 0


def _stop_tqdm_monitor() -> None:
    try:
        from tqdm.auto import tqdm
        from tqdm.std import tqdm as std_tqdm
    except ImportError:
        return
    for tqdm_cls in (tqdm, std_tqdm):
        monitor = getattr(tqdm_cls, "monitor", None)
        if monitor is not None:
            monitor.exit()
            tqdm_cls.monitor = None


def _normalize_row(dataset: str, row: Mapping[str, object], row_index: int) -> dict[str, object] | None:
    if dataset == "HuggingFaceM4/DocumentVQA":
        return _vqa(row.get("image"), row.get("question"), _first(row.get("answers")))
    if dataset == "lmms-lab/textvqa":
        return _vqa(row.get("image"), row.get("question"), _majority(row.get("answers")), " ".join(row.get("ocr_tokens") or []))
    if dataset == "howard-hou/OCR-VQA":
        questions = row.get("questions") or []
        answers = row.get("answers") or []
        if not questions or not answers:
            return None
        index = row_index % min(len(questions), len(answers))
        return _vqa(row.get("image"), str(questions[index]), str(answers[index]), " ".join(row.get("ocr_tokens") or []))
    if dataset == "HuggingFaceM4/ChartQA":
        return _vqa(row.get("image"), row.get("query"), _first(row.get("label")))
    return None


def _vqa(image, question, answer, ocr_text: str = "") -> dict[str, object] | None:
    if image is None or not question or not answer:
        return None
    return {
        "image": image,
        "question": str(question),
        "answer": str(answer),
        "ocr_text": ocr_text,
    }


def _first(value) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, tuple) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def _majority(value) -> str:
    if isinstance(value, list) and value:
        return str(Counter(str(item) for item in value).most_common(1)[0][0])
    return _first(value)


def _save_image(output_dir: Path, split: str, index: int, image) -> Path:
    from PIL import Image

    split_dir = output_dir / split / "images"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{index:05d}.png"
    if isinstance(image, Image.Image):
        converted = image.convert("RGB")
        converted.save(path)
        converted.close()
        image.close()
        return path
    if hasattr(image, "convert"):
        converted = image.convert("RGB")
        converted.save(path)
        converted.close()
        close = getattr(image, "close", None)
        if close is not None:
            close()
        return path
    raise ValueError(f"unsupported image type: {type(image)!r}")


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")
