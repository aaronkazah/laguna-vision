import json

from lagunavision.data.manifest import load_manifest
from lagunavision.eval.capability_probe import generate_capability_probe


def test_capability_probe_generates_expected_pass_and_failure_categories(tmp_path) -> None:
    manifest = generate_capability_probe(tmp_path)
    items = load_manifest(manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

    assert len(items) == 80
    assert all(item.image.exists() for item in items)
    assert {row["expected_result"] for row in rows} == {"measure"}
    assert sum(row["category"] == "basic_shape" for row in rows) == 10
    assert sum(row["category"] == "basic_color" for row in rows) == 10
    assert sum(row["category"] == "color_shape_binding" for row in rows) == 10
    assert sum(row["category"] == "no_text_control" for row in rows) == 10
    assert sum(row["category"] == "tiny_ocr" for row in rows) == 10
    assert sum(row["category"] == "dense_ui_localization" for row in rows) == 10
    assert sum(row["category"] == "meme_semantics" for row in rows) == 10
    assert sum(row["category"] == "table_precision" for row in rows) == 10
    assert any(row["must_not_include"] for row in rows if row["category"] == "basic_color")
    assert (tmp_path / "summary.json").exists()
