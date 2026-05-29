"""Synthetic image-only fixtures for proving visual conditioning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

QUESTION = "What exact colored shape is shown? Answer with the color and shape only."
IMAGE_SIZE = 384


@dataclass(frozen=True)
class ShapeSpec:
    color_name: str
    shape_name: str
    rgb: tuple[int, int, int]

    @property
    def answer(self) -> str:
        return f"{self.color_name} {self.shape_name}"


SHAPES: tuple[ShapeSpec, ...] = (
    ShapeSpec("red", "circle", (220, 38, 38)),
    ShapeSpec("blue", "square", (37, 99, 235)),
    ShapeSpec("green", "triangle", (22, 163, 74)),
    ShapeSpec("yellow", "star", (234, 179, 8)),
    ShapeSpec("purple", "diamond", (147, 51, 234)),
    ShapeSpec("orange", "hexagon", (249, 115, 22)),
    ShapeSpec("black", "cross", (24, 24, 27)),
    ShapeSpec("cyan", "ring", (6, 182, 212)),
)


def write_visual_overfit_dataset(output_dir: Path, train_count: int = 32, eval_count: int = 16) -> None:
    if train_count < len(SHAPES):
        raise ValueError(f"train_count must be at least {len(SHAPES)}")
    if eval_count < len(SHAPES):
        raise ValueError(f"eval_count must be at least {len(SHAPES)}")
    if eval_count > train_count:
        raise ValueError("eval_count must not exceed train_count")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    blank_image = image_dir / "blank.png"
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (245, 245, 245)).save(blank_image)

    train_rows = [_row(image_dir, index) for index in range(train_count)]
    eval_rows = train_rows[:eval_count]
    wrong_rows = [
        {**row, "image": eval_rows[(index + 1) % eval_count]["image"]}
        for index, row in enumerate(eval_rows)
    ]
    blank_rows = [{**row, "image": str(blank_image)} for row in eval_rows]

    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "eval.jsonl", eval_rows)
    _write_jsonl(output_dir / "wrong.jsonl", wrong_rows)
    _write_jsonl(output_dir / "blank.jsonl", blank_rows)


def _row(image_dir: Path, index: int) -> dict[str, object]:
    spec = SHAPES[index % len(SHAPES)]
    variant = index // len(SHAPES)
    image_path = image_dir / f"{index:04d}_{spec.color_name}_{spec.shape_name}.png"
    _draw_shape(image_path, spec, variant)
    return {
        "id": f"visual-overfit-{index:04d}",
        "image": str(image_path),
        "question": QUESTION,
        "answer": spec.answer,
        "must_include": [spec.answer],
        "accepted_fix_terms": [],
        "ocr_text": "",
        "rubric": "vqa",
        "source": "visual_overfit",
    }


def _draw_shape(path: Path, spec: ShapeSpec, variant: int) -> None:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (248, 250, 252))
    draw = ImageDraw.Draw(image)

    offset_x = ((variant * 37) % 80) - 40
    offset_y = ((variant * 53) % 80) - 40
    radius = 108 + ((variant * 11) % 24)
    cx = IMAGE_SIZE // 2 + offset_x
    cy = IMAGE_SIZE // 2 + offset_y
    box = [cx - radius, cy - radius, cx + radius, cy + radius]

    if spec.shape_name == "circle":
        draw.ellipse(box, fill=spec.rgb)
    elif spec.shape_name == "square":
        draw.rectangle(box, fill=spec.rgb)
    elif spec.shape_name == "triangle":
        draw.polygon([(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)], fill=spec.rgb)
    elif spec.shape_name == "star":
        draw.polygon(_star_points(cx, cy, radius, radius // 2), fill=spec.rgb)
    elif spec.shape_name == "diamond":
        draw.polygon([(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)], fill=spec.rgb)
    elif spec.shape_name == "hexagon":
        draw.polygon(_regular_polygon(cx, cy, radius, 6), fill=spec.rgb)
    elif spec.shape_name == "cross":
        width = radius // 2
        draw.rectangle([cx - width, cy - radius, cx + width, cy + radius], fill=spec.rgb)
        draw.rectangle([cx - radius, cy - width, cx + radius, cy + width], fill=spec.rgb)
    elif spec.shape_name == "ring":
        draw.ellipse(box, fill=spec.rgb)
        inset = radius // 2
        draw.ellipse([cx - inset, cy - inset, cx + inset, cy + inset], fill=(248, 250, 252))
    else:  # pragma: no cover
        raise ValueError(f"unsupported shape: {spec.shape_name}")

    image.save(path)


def _regular_polygon(cx: int, cy: int, radius: int, sides: int) -> list[tuple[float, float]]:
    return [
        (
            cx + radius * math.cos((math.tau * point / sides) - math.pi / 2),
            cy + radius * math.sin((math.tau * point / sides) - math.pi / 2),
        )
        for point in range(sides)
    ]


def _star_points(cx: int, cy: int, outer: int, inner: int) -> list[tuple[float, float]]:
    return [
        (
            cx + (outer if point % 2 == 0 else inner) * math.cos((math.tau * point / 10) - math.pi / 2),
            cy + (outer if point % 2 == 0 else inner) * math.sin((math.tau * point / 10) - math.pi / 2),
        )
        for point in range(10)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
