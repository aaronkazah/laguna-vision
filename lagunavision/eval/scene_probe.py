from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneProbeCase:
    id: str
    question: str
    must_include: tuple[str, ...]
    accepted_terms: tuple[str, ...]


SCENE_PROBES = (
    SceneProbeCase(
        id="scene_city_bus_001",
        question="Explain what is going on in this scene.",
        must_include=("bus", "cyclist"),
        accepted_terms=("street", "city", "traffic", "road"),
    ),
    SceneProbeCase(
        id="scene_space_rover_002",
        question="Explain what is happening in this image.",
        must_include=("rover", "planet"),
        accepted_terms=("space", "mars", "robot", "rocks"),
    ),
    SceneProbeCase(
        id="scene_kitchen_alarm_003",
        question="What situation is shown here?",
        must_include=("kitchen", "steam"),
        accepted_terms=("alarm", "pan", "cooking", "warning"),
    ),
    SceneProbeCase(
        id="scene_dashboard_004",
        question="What is this screen showing?",
        must_include=("chart", "dashboard"),
        accepted_terms=("metrics", "graph", "analytics", "data"),
    ),
    SceneProbeCase(
        id="scene_park_dog_005",
        question="Describe the activity in this image.",
        must_include=("dog", "park"),
        accepted_terms=("ball", "tree", "playing", "grass"),
    ),
)


def generate_scene_probe(output_dir: Path, limit: int = 5) -> Path:
    if limit <= 0:
        raise ValueError("limit must be positive")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    cases = SCENE_PROBES[:limit]
    with manifest.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            image_name = f"{index:03d}.png"
            _draw_scene(images_dir / image_name, case.id, variant=index)
            _write_case(handle, case, f"images/{image_name}", case.id)
    return manifest


def generate_scene_dataset(
    output_dir: Path,
    train_count: int = 40,
    eval_count: int = 10,
    seed: int = 7,
) -> tuple[Path, Path]:
    if train_count <= 0 or eval_count <= 0:
        raise ValueError("train_count and eval_count must be positive")
    rng = random.Random(seed)
    train_manifest = _generate_scene_split(output_dir, "train", train_count, rng)
    eval_manifest = _generate_scene_split(output_dir, "eval", eval_count, rng)
    return train_manifest, eval_manifest


def _generate_scene_split(output_dir: Path, split: str, count: int, rng: random.Random) -> Path:
    images_dir = output_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f"{split}.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index in range(count):
            case = SCENE_PROBES[index % len(SCENE_PROBES)]
            variant = rng.randint(1, 1_000_000)
            item_id = f"{case.id}_{split}_{index:04d}"
            image_name = f"{item_id}.png"
            _draw_scene(images_dir / image_name, case.id, variant=variant)
            _write_case(handle, case, f"{split}/images/{image_name}", item_id)
    return manifest


def _write_case(handle, case: SceneProbeCase, image: str, item_id: str) -> None:
    handle.write(
        json.dumps(
            {
                "id": item_id,
                "image": image,
                "question": case.question,
                "ocr_text": "",
                "rubric": "description",
                "must_include": case.must_include,
                "accepted_fix_terms": case.accepted_terms,
                "must_not_include": (),
            },
            sort_keys=True,
        )
        + "\n"
    )


def _draw_scene(path: Path, scene_id: str, variant: int = 0) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Install dev dependencies with `python -m pip install -e '.[dev]'`.") from exc

    rng = random.Random(variant)
    image = Image.new("RGB", (960, 640), "#f8fafc")
    draw = ImageDraw.Draw(image)
    if scene_id == "scene_city_bus_001":
        _city_bus(draw)
    elif scene_id == "scene_space_rover_002":
        _space_rover(draw)
    elif scene_id == "scene_kitchen_alarm_003":
        _kitchen_alarm(draw)
    elif scene_id == "scene_dashboard_004":
        _dashboard(draw)
    else:
        _park_dog(draw)
    for _ in range(8):
        x = rng.randint(0, 920)
        y = rng.randint(0, 600)
        color = rng.choice(("#e2e8f0", "#cbd5e1", "#fde68a", "#bfdbfe", "#bbf7d0"))
        draw.ellipse((x, y, x + rng.randint(8, 30), y + rng.randint(8, 30)), fill=color)
    image.save(path)


def _city_bus(draw) -> None:
    draw.rectangle((0, 420, 960, 640), fill="#334155")
    draw.rectangle((80, 260, 520, 410), fill="#facc15", outline="#0f172a", width=4)
    draw.text((230, 305), "CITY BUS", fill="#0f172a")
    draw.ellipse((130, 380, 190, 440), fill="#0f172a")
    draw.ellipse((410, 380, 470, 440), fill="#0f172a")
    draw.ellipse((680, 360, 735, 415), outline="#0f172a", width=5)
    draw.ellipse((785, 360, 840, 415), outline="#0f172a", width=5)
    draw.line((707, 360, 760, 310, 812, 360), fill="#0f172a", width=5)
    draw.text((700, 280), "cyclist", fill="#0f172a")


def _space_rover(draw) -> None:
    draw.rectangle((0, 0, 960, 430), fill="#111827")
    draw.rectangle((0, 430, 960, 640), fill="#b45309")
    draw.ellipse((720, 80, 840, 200), fill="#f8fafc")
    draw.rectangle((330, 360, 610, 450), fill="#94a3b8", outline="#0f172a", width=4)
    draw.text((405, 388), "ROVER", fill="#0f172a")
    draw.ellipse((350, 430, 420, 500), fill="#0f172a")
    draw.ellipse((520, 430, 590, 500), fill="#0f172a")
    draw.line((470, 360, 560, 280), fill="#94a3b8", width=8)
    draw.ellipse((550, 250, 620, 320), fill="#38bdf8")
    draw.text((80, 490), "red planet rocks", fill="#fef3c7")


def _kitchen_alarm(draw) -> None:
    draw.rectangle((0, 0, 960, 640), fill="#fde68a")
    draw.rectangle((80, 380, 880, 560), fill="#92400e")
    draw.rectangle((340, 300, 620, 380), fill="#334155")
    draw.text((420, 325), "PAN", fill="#f8fafc")
    draw.ellipse((410, 190, 550, 290), outline="#64748b", width=8)
    draw.ellipse((390, 130, 570, 260), outline="#94a3b8", width=8)
    draw.rectangle((730, 90, 860, 180), fill="#ef4444")
    draw.text((750, 125), "SMOKE\nALARM", fill="#ffffff")


def _dashboard(draw) -> None:
    draw.rectangle((40, 40, 920, 600), fill="#0f172a")
    draw.text((80, 80), "Analytics Dashboard", fill="#f8fafc")
    draw.rectangle((90, 150, 460, 520), outline="#38bdf8", width=3)
    draw.line((120, 470, 180, 400, 250, 430, 320, 260, 420, 210), fill="#22c55e", width=6)
    draw.text((140, 170), "chart", fill="#f8fafc")
    draw.rectangle((540, 160, 840, 230), fill="#1d4ed8")
    draw.rectangle((540, 270, 790, 340), fill="#7c3aed")
    draw.rectangle((540, 380, 870, 450), fill="#ea580c")
    draw.text((600, 500), "metrics", fill="#f8fafc")


def _park_dog(draw) -> None:
    draw.rectangle((0, 0, 960, 420), fill="#bfdbfe")
    draw.rectangle((0, 420, 960, 640), fill="#22c55e")
    draw.rectangle((120, 240, 170, 460), fill="#854d0e")
    draw.ellipse((60, 120, 230, 300), fill="#16a34a")
    draw.ellipse((430, 390, 620, 500), fill="#a16207")
    draw.ellipse((580, 350, 690, 430), fill="#a16207")
    draw.ellipse((650, 375, 675, 400), fill="#0f172a")
    draw.ellipse((760, 420, 820, 480), fill="#ef4444")
    draw.text((410, 520), "dog in park", fill="#0f172a")
