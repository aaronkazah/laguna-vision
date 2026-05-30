from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


WORDS = (
    "alpha",
    "bravo",
    "cedar",
    "delta",
    "ember",
    "falcon",
    "green",
    "harbor",
    "indigo",
    "jasper",
)


@dataclass(frozen=True)
class SpatialOcrExample:
    id: str
    image: str
    question: str
    answer: str
    labels: dict[str, str]


def generate_spatial_ocr_manifest(output_dir: Path, count: int, seed: int = 7) -> tuple[SpatialOcrExample, ...]:
    if count <= 0:
        raise ValueError("count must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)
    examples = tuple(_example(index, rng) for index in range(count))
    for example in examples:
        _write_image(images_dir / Path(example.image).name, example)

    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item.__dict__, sort_keys=True) + "\n")
    return examples


def _example(index: int, rng: random.Random) -> SpatialOcrExample:
    regions = ("top left", "top right", "bottom left", "bottom right")
    words = rng.sample(WORDS, 4)
    selected = rng.randrange(4)
    labels = dict(zip(regions, words))
    return SpatialOcrExample(
        id=f"spatial_ocr_{index:04d}",
        image=f"images/spatial_ocr_{index:04d}.png",
        question=f"What word is in the {regions[selected]}?",
        answer=labels[regions[selected]],
        labels=labels,
    )


def _write_image(path: Path, example: SpatialOcrExample) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Install data dependencies with `python -m pip install -e '.[data]'`.") from exc

    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    boxes = {
        "top left": (24, 24, 296, 216),
        "top right": (344, 24, 616, 216),
        "bottom left": (24, 264, 296, 456),
        "bottom right": (344, 264, 616, 456),
    }
    for region, box in boxes.items():
        draw.rectangle(box, outline="black", width=3)
        draw.text((box[0] + 24, box[1] + 72), example.labels[region], fill="black")
    image.save(path)
