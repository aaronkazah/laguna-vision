from __future__ import annotations

import json

from PIL import Image

from lagunavision.data.llava import materialize_llava_json
from lagunavision.data.manifest import load_manifest
from lagunavision.train.visual_bridge import _target_answer


def test_materialize_llava_json_conversations(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), "red").save(image_dir / "sample.jpg")
    source = tmp_path / "llava.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "image": "sample.jpg",
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "A red square is shown."},
                        {"from": "human", "value": "What color is it?"},
                        {"from": "gpt", "value": "It is red."},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = materialize_llava_json(
        source,
        tmp_path / "manifest",
        image_roots=[image_dir],
        image_mode="copy",
        eval_count=1,
    )

    assert result.eval_count == 1
    assert result.train_count == 1
    eval_items = load_manifest(result.eval_manifest)
    train_items = load_manifest(result.train_manifest)
    assert eval_items[0].answer == "A red square is shown."
    assert train_items[0].answer == "It is red."
    assert "What color is it?" in train_items[0].question
    assert train_items[0].image.exists()
    assert _target_answer(train_items[0]) == "It is red."
