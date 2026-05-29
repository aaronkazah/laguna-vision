from lagunavision.data.manifest import load_manifest
from lagunavision.eval.scene_probe import generate_scene_probe


def test_scene_probe_generates_original_images(tmp_path) -> None:
    manifest = generate_scene_probe(tmp_path, limit=3)
    items = load_manifest(manifest)

    assert len(items) == 3
    assert all(item.image.exists() for item in items)
    assert all(item.rubric == "description" for item in items)
