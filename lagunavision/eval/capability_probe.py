from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


COLOR_HEX = {
    "red": "#dc2626",
    "blue": "#2563eb",
    "green": "#16a34a",
    "yellow": "#facc15",
    "purple": "#9333ea",
    "orange": "#f97316",
    "pink": "#ec4899",
    "cyan": "#06b6d4",
    "black": "#111827",
    "brown": "#92400e",
}
COLORS = tuple(COLOR_HEX)
SHAPES = ("circle", "square", "triangle", "star", "diamond", "rectangle", "oval", "pentagon", "hexagon", "cross")
SHAPE_COLOR_PAIRS = (
    ("red", "circle"),
    ("blue", "square"),
    ("green", "triangle"),
    ("yellow", "star"),
    ("purple", "diamond"),
    ("orange", "rectangle"),
    ("pink", "oval"),
    ("cyan", "pentagon"),
    ("black", "hexagon"),
    ("brown", "cross"),
)


@dataclass(frozen=True)
class CapabilityProbeCase:
    id: str
    title: str
    category: str
    expected_result: str
    question: str
    expected_answer: str
    must_include: tuple[str, ...]
    accepted_terms: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    must_include_mode: str = "any"
    max_answer_words: int = 0
    rubric: str = "vqa"
    notes: str = ""
    image_kind: str = "shape"
    image_params: dict[str, str] = field(default_factory=dict)


def _shape_case(index: int, color: str, shape: str) -> CapabilityProbeCase:
    return CapabilityProbeCase(
        id=f"shape_{index:02d}_{color}_{shape}",
        title=f"Shape only: {color} {shape}",
        category="basic_shape",
        expected_result="measure",
        question="What shape is shown? Give the final answer as exactly one word.",
        expected_answer=shape,
        must_include=(shape,),
        must_not_include=tuple(other for other in SHAPES if other != shape),
        max_answer_words=1,
        notes="Separates shape recognition from color recognition.",
        image_kind="shape",
        image_params={"color": color, "shape": shape},
    )


def _color_case(index: int, color: str, shape: str) -> CapabilityProbeCase:
    return CapabilityProbeCase(
        id=f"color_{index:02d}_{color}_{shape}",
        title=f"Color only: {color} {shape}",
        category="basic_color",
        expected_result="measure",
        question="What color is the shape? Give the final answer as exactly one word.",
        expected_answer=color,
        must_include=(color,),
        must_not_include=tuple(other for other in COLORS if other != color),
        max_answer_words=1,
        notes="Separates color recognition from shape recognition.",
        image_kind="shape",
        image_params={"color": color, "shape": shape},
    )


def _binding_case(index: int, color: str, shape: str) -> CapabilityProbeCase:
    return CapabilityProbeCase(
        id=f"binding_{index:02d}_{color}_{shape}",
        title=f"Color and shape: {color} {shape}",
        category="color_shape_binding",
        expected_result="measure",
        question="What exact colored shape is shown? Give the final answer as exactly two words: color shape.",
        expected_answer=f"{color} {shape}",
        must_include=(color, shape),
        must_not_include=tuple(other for other in COLORS if other != color)
        + tuple(other for other in SHAPES if other != shape),
        must_include_mode="all",
        max_answer_words=2,
        notes="Checks whether the model can bind the color to the shape.",
        image_kind="shape",
        image_params={"color": color, "shape": shape},
    )


def _no_text_case(index: int) -> CapabilityProbeCase:
    return CapabilityProbeCase(
        id=f"no_text_{index:02d}",
        title=f"No visible text control {index}",
        category="no_text_control",
        expected_result="measure",
        question="Is there visible text? Give the final answer as exactly one word: yes or no.",
        expected_answer="no",
        must_include=("no",),
        must_not_include=("yes", "hello", "submit", "error", "revenue"),
        max_answer_words=1,
        notes="Checks that the model can abstain instead of inventing OCR.",
        image_kind="no_text",
        image_params={"index": str(index)},
    )


OCR_PHRASES = (
    "forkpty: Resource temporarily unavailable",
    "CUDA out of memory",
    "ModuleNotFoundError: lagunavision",
    "Permission denied",
    "Connection timed out",
    "No such file or directory",
    "RuntimeError: checkpoint missing",
    "ValueError: invalid shape",
    "HTTP 401 Unauthorized",
    "FileNotFoundError: config.json",
)


def _ocr_case(index: int, phrase: str) -> CapabilityProbeCase:
    return CapabilityProbeCase(
        id=f"tiny_ocr_{index:02d}",
        title=f"Tiny terminal OCR {index}",
        category="tiny_ocr",
        expected_result="measure",
        question="Copy only the exact terminal error phrase as the final answer.",
        expected_answer=phrase,
        must_include=(phrase,),
        must_include_mode="all",
        max_answer_words=max(4, len(phrase.replace(":", " ").split()) + 1),
        notes="Small terminal text; current checkpoint is weak at exact OCR.",
        image_kind="terminal",
        image_params={"phrase": phrase, "index": str(index)},
    )


UI_CASES = (
    ("deploy", "red"),
    ("cache", "orange"),
    ("train", "red"),
    ("eval", "purple"),
    ("upload", "red"),
    ("serve", "orange"),
    ("index", "red"),
    ("package", "purple"),
    ("test", "red"),
    ("materialize", "orange"),
)
UI_JOBS = tuple(job for job, _ in UI_CASES)


def _ui_case(index: int, job: str, color: str) -> CapabilityProbeCase:
    return CapabilityProbeCase(
        id=f"dense_ui_{index:02d}_{job}_{color}",
        title=f"Dense UI failed row: {job} {color}",
        category="dense_ui_localization",
        expected_result="measure",
        question="Which job row failed, and what color is its status pill? Give the final answer as exactly two words: job color.",
        expected_answer=f"{job} {color}",
        must_include=(job, color),
        must_not_include=tuple(other for other in UI_JOBS if other != job)
        + tuple(other for other in ("red", "orange", "purple", "green") if other != color),
        must_include_mode="all",
        max_answer_words=2,
        notes="Requires reading a small UI table and localizing the failed row.",
        image_kind="ui",
        image_params={"job": job, "color": color},
    )


MEME_DIRECTIONS = ("left", "right", "both", "unclear", "left", "right", "both", "unclear", "left", "both")


def _meme_case(index: int, direction: str) -> CapabilityProbeCase:
    return CapabilityProbeCase(
        id=f"meme_{index:02d}_{direction}",
        title=f"Meme attribution: {direction}",
        category="meme_semantics",
        expected_result="measure",
        question="Which side is pushing the center character? Give the final answer as exactly one word: left, right, both, or unclear.",
        expected_answer=direction,
        must_include=(direction,),
        must_not_include=tuple(other for other in ("left", "right", "both", "unclear") if other != direction),
        max_answer_words=1,
        notes="Meme images need visual attribution; reasoning is allowed, but scoring uses the final answer.",
        image_kind="meme",
        image_params={"direction": direction, "index": str(index)},
    )


TABLE_VALUES = {
    ("revenue", "Q1"): "$3.2M",
    ("revenue", "Q2"): "$4.7M",
    ("revenue", "Q3"): "$4.1M",
    ("revenue", "Q4"): "$5.0M",
    ("margin", "Q1"): "18%",
    ("margin", "Q2"): "21%",
    ("margin", "Q3"): "19%",
    ("users", "Q1"): "42k",
    ("users", "Q2"): "57k",
    ("latency", "Q3"): "128ms",
}


def _table_case(index: int, metric: str, quarter: str, value: str) -> CapabilityProbeCase:
    metric_label = metric.capitalize()
    return CapabilityProbeCase(
        id=f"table_{index:02d}_{metric}_{quarter.casefold()}",
        title=f"Table value: {metric_label} {quarter}",
        category="table_precision",
        expected_result="measure",
        question=f"What is the {quarter} {metric} value in the table? Give the final answer as only the value.",
        expected_answer=value,
        must_include=(value, value.replace("$", "")),
        max_answer_words=1,
        notes="Precise table extraction remains unreliable.",
        image_kind="table",
        image_params={"metric": metric, "quarter": quarter},
    )


CAPABILITY_PROBES = (
    *(_shape_case(index, color, shape) for index, (color, shape) in enumerate(SHAPE_COLOR_PAIRS, start=1)),
    *(_color_case(index, color, shape) for index, (color, shape) in enumerate(SHAPE_COLOR_PAIRS, start=1)),
    *(_binding_case(index, color, shape) for index, (color, shape) in enumerate(SHAPE_COLOR_PAIRS, start=1)),
    *(_no_text_case(index) for index in range(1, 11)),
    *(_ocr_case(index, phrase) for index, phrase in enumerate(OCR_PHRASES, start=1)),
    *(_ui_case(index, job, color) for index, (job, color) in enumerate(UI_CASES, start=1)),
    *(_meme_case(index, direction) for index, direction in enumerate(MEME_DIRECTIONS, start=1)),
    *(_table_case(index, metric, quarter, value) for index, ((metric, quarter), value) in enumerate(TABLE_VALUES.items(), start=1)),
)


def generate_capability_probe(output_dir: Path) -> Path:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(CAPABILITY_PROBES, start=1):
            image_name = f"{index:03d}_{case.id}.png"
            _write_probe_image(images_dir / image_name, case)
            handle.write(json.dumps(_manifest_row(case, f"images/{image_name}"), sort_keys=True) + "\n")

    summary_path = output_dir / "summary.json"
    by_category: dict[str, int] = {}
    for case in CAPABILITY_PROBES:
        by_category[case.category] = by_category.get(case.category, 0) + 1
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "items": len(CAPABILITY_PROBES),
                "by_category": by_category,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _manifest_row(case: CapabilityProbeCase, image: str) -> dict[str, object]:
    return {
        "id": case.id,
        "image": image,
        "question": case.question,
        "answer": case.expected_answer,
        "ocr_text": "",
        "rubric": case.rubric,
        "must_include": case.must_include,
        "must_include_mode": case.must_include_mode,
        "max_answer_words": case.max_answer_words,
        "accepted_fix_terms": case.accepted_terms,
        "must_not_include": case.must_not_include,
        "category": case.category,
        "expected_result": case.expected_result,
        "notes": case.notes,
    }


def _write_probe_image(path: Path, case: CapabilityProbeCase) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Install dev dependencies with `python -m pip install -e '.[dev]'`.") from exc

    image = Image.new("RGB", (960, 640), "#f8fafc")
    draw = ImageDraw.Draw(image)

    if case.image_kind == "shape":
        _draw_colored_shape(draw, case.image_params["color"], case.image_params["shape"])
    elif case.image_kind == "no_text":
        _draw_no_text(draw, int(case.image_params["index"]))
    elif case.image_kind == "terminal":
        _draw_tiny_terminal(draw, case.image_params["phrase"], int(case.image_params["index"]))
    elif case.image_kind == "ui":
        _draw_dense_ui(draw, case.image_params["job"], case.image_params["color"])
    elif case.image_kind == "meme":
        _draw_meme(draw, case.image_params["direction"], int(case.image_params["index"]))
    elif case.image_kind == "table":
        _draw_document_table(draw)
    else:  # pragma: no cover
        raise ValueError(f"unsupported capability probe image kind: {case.image_kind}")

    image.save(path)


def _draw_colored_shape(draw, color: str, shape: str) -> None:
    fill = COLOR_HEX[color]
    if shape == "circle":
        draw.ellipse((270, 110, 690, 530), fill=fill)
    elif shape == "square":
        draw.rectangle((270, 110, 690, 530), fill=fill)
    elif shape == "triangle":
        draw.polygon([(480, 100), (250, 540), (710, 540)], fill=fill)
    elif shape == "star":
        points = [
            (480, 90),
            (545, 280),
            (745, 280),
            (585, 395),
            (650, 585),
            (480, 470),
            (310, 585),
            (375, 395),
            (215, 280),
            (415, 280),
        ]
        draw.polygon(points, fill=fill)
    elif shape == "diamond":
        draw.polygon([(480, 90), (740, 320), (480, 550), (220, 320)], fill=fill)
    elif shape == "rectangle":
        draw.rectangle((200, 180, 760, 460), fill=fill)
    elif shape == "oval":
        draw.ellipse((180, 190, 780, 450), fill=fill)
    elif shape == "pentagon":
        draw.polygon([(480, 80), (735, 270), (635, 555), (325, 555), (225, 270)], fill=fill)
    elif shape == "hexagon":
        draw.polygon([(350, 110), (610, 110), (760, 320), (610, 530), (350, 530), (200, 320)], fill=fill)
    elif shape == "cross":
        draw.rectangle((390, 120, 570, 520), fill=fill)
        draw.rectangle((230, 270, 730, 370), fill=fill)
    else:  # pragma: no cover
        raise ValueError(f"unsupported shape: {shape}")


def _draw_no_text(draw, index: int) -> None:
    color_a = COLORS[(index - 1) % len(COLORS)]
    color_b = COLORS[(index + 3) % len(COLORS)]
    shape_a = SHAPES[(index - 1) % len(SHAPES)]
    shape_b = SHAPES[(index + 4) % len(SHAPES)]
    draw.rectangle((60, 60, 900, 580), fill="#e2e8f0")
    _draw_small_shape(draw, shape_a, COLOR_HEX[color_a], (100, 120, 430, 500))
    _draw_small_shape(draw, shape_b, COLOR_HEX[color_b], (520, 120, 860, 500))


def _draw_small_shape(draw, shape: str, fill: str, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    if shape in {"circle", "oval"}:
        draw.ellipse(box, fill=fill)
    elif shape in {"square", "rectangle"}:
        draw.rectangle(box, fill=fill)
    elif shape == "triangle":
        draw.polygon([(cx, top), (left, bottom), (right, bottom)], fill=fill)
    elif shape == "diamond":
        draw.polygon([(cx, top), (right, cy), (cx, bottom), (left, cy)], fill=fill)
    elif shape == "cross":
        draw.rectangle((cx - 45, top, cx + 45, bottom), fill=fill)
        draw.rectangle((left, cy - 45, right, cy + 45), fill=fill)
    else:
        draw.polygon([(cx, top), (right, top + 120), (right - 50, bottom), (left + 50, bottom), (left, top + 120)], fill=fill)


def _draw_tiny_terminal(draw, phrase: str, index: int) -> None:
    draw.rectangle((60, 60, 900, 580), fill="#0f172a")
    draw.rectangle((60, 60, 900, 100), fill="#111827")
    draw.text((80, 76), "terminal", fill="#94a3b8")
    lines = [
        f"$ laguna probe run --case {index:02d}",
        phrase,
        "diagnostics saved to /tmp/laguna-probe.log",
        "retry after reading the exact error line",
    ]
    y = 160
    for line in lines:
        draw.text((90, y), line, fill="#e5e7eb")
        y += 24


def _draw_dense_ui(draw, target_job: str, target_color: str) -> None:
    status_hex = {"red": "#ef4444", "orange": "#f97316", "purple": "#a855f7"}[target_color]
    draw.rectangle((40, 40, 920, 600), fill="#111827")
    draw.text((80, 74), "Laguna run dashboard", fill="#f9fafb")
    draw.text((90, 112), "job", fill="#cbd5e1")
    draw.text((500, 112), "status", fill="#cbd5e1")
    y = 140
    for job in UI_JOBS:
        failed = job == target_job
        label = "failed" if failed else "ok"
        color = status_hex if failed else "#22c55e"
        draw.rectangle((80, y, 860, y + 38), outline="#334155", width=2)
        draw.text((110, y + 13), job, fill="#f8fafc")
        draw.rounded_rectangle((500, y + 8, 650, y + 30), radius=11, fill=color)
        draw.text((535, y + 14), label, fill="#ffffff" if failed else "#111827")
        y += 43


def _draw_meme(draw, direction: str, index: int) -> None:
    draw.rectangle((0, 0, 960, 640), fill="#fde68a")
    draw.ellipse((430, 210, 530, 310), fill="#fbbf24", outline="#0f172a", width=3)
    draw.line((480, 310, 480, 470), fill="#0f172a", width=6)
    draw.line((480, 360, 350, 440), fill="#0f172a", width=6)
    draw.line((480, 360, 610, 440), fill="#0f172a", width=6)
    draw.line((480, 470, 410, 560), fill="#0f172a", width=6)
    draw.line((480, 470, 560, 560), fill="#0f172a", width=6)
    draw.ellipse((160, 230, 250, 320), fill="#bfdbfe", outline="#0f172a", width=3)
    draw.ellipse((705, 230, 795, 320), fill="#fecaca", outline="#0f172a", width=3)
    if direction in {"left", "both"}:
        draw.line((250, 290, 415, 340), fill="#dc2626", width=12)
        draw.polygon([(415, 340), (380, 315), (385, 365)], fill="#dc2626")
    if direction in {"right", "both"}:
        draw.line((705, 290, 545, 340), fill="#2563eb", width=12)
        draw.polygon([(545, 340), (580, 315), (575, 365)], fill="#2563eb")
    if direction == "unclear":
        draw.arc((310, 250, 650, 520), start=20 + index, end=160 + index, fill="#64748b", width=8)


def _draw_document_table(draw) -> None:
    draw.rectangle((100, 60, 860, 585), fill="#ffffff", outline="#cbd5e1", width=3)
    draw.text((145, 105), "Quarterly Metrics", fill="#111827")
    x0, y0 = 145, 175
    cell_w, cell_h = 135, 58
    rows = (
        ("Metric", "Q1", "Q2", "Q3", "Q4"),
        ("Revenue", "$3.2M", "$4.7M", "$4.1M", "$5.0M"),
        ("Margin", "18%", "21%", "19%", "23%"),
        ("Users", "42k", "57k", "63k", "71k"),
        ("Latency", "142ms", "136ms", "128ms", "119ms"),
    )
    for row_index, row in enumerate(rows):
        for col_index, text in enumerate(row):
            left = x0 + col_index * cell_w
            top = y0 + row_index * cell_h
            fill = "#f1f5f9" if row_index == 0 or col_index == 0 else "#ffffff"
            draw.rectangle((left, top, left + cell_w, top + cell_h), fill=fill, outline="#94a3b8", width=2)
            draw.text((left + 14, top + 21), text, fill="#111827")
