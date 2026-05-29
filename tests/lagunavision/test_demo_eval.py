from lagunavision.data.manifest import load_manifest
from lagunavision.eval.demo_set import generate_demo_eval


def test_demo_eval_generates_15_manifest_items_and_images(tmp_path) -> None:
    manifest = generate_demo_eval(tmp_path)
    items = load_manifest(manifest)

    assert len(items) == 15
    assert all(item.image.exists() for item in items)
    assert all(item.ocr_text for item in items)
