"""Лёгкие hygiene-проверки для EDA notebooks."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks/eda")


def test_eda_notebooks_have_unique_cell_ids() -> None:
    """Все EDA notebooks должны иметь уникальные cell id в сыром JSON."""
    missing_ids: dict[str, list[int]] = {}
    duplicate_ids: dict[str, list[str]] = {}

    for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cells = data.get("cells", [])
        seen_ids: set[str] = set()
        notebook_duplicates: set[str] = set()

        for index, cell in enumerate(cells):
            cell_id = cell.get("id")
            if not cell_id:
                missing_ids.setdefault(str(path), []).append(index)
                continue
            if cell_id in seen_ids:
                notebook_duplicates.add(cell_id)
            seen_ids.add(cell_id)

        if notebook_duplicates:
            duplicate_ids[str(path)] = sorted(notebook_duplicates)

    assert missing_ids == {}
    assert duplicate_ids == {}
