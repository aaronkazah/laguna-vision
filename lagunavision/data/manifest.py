from __future__ import annotations

import json
from pathlib import Path

from lagunavision.types import EvalManifestItem


def load_manifest(path: Path) -> tuple[EvalManifestItem, ...]:
    root = path.parent
    items: list[EvalManifestItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            try:
                items.append(EvalManifestItem.from_mapping(row, root))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return tuple(items)

