# Тестовый файл `test_analysis_notebooks.py` домена `notebooks`.
#
# Этот файл проверяет только:
# - проверку логики домена: активные notebook и их smoke-проверки;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `notebooks` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import json
from pathlib import Path


def _iter_active_notebook_paths() -> list[Path]:
    # Активные notebook теперь разложены по рабочим каталогам.
    notebook_roots = (
        Path("analysis/notebooks/eda"),
        Path("analysis/notebooks/research"),
        Path("analysis/notebooks/technical"),
    )
    notebook_paths: list[Path] = []
    for notebook_root in notebook_roots:
        notebook_paths.extend(sorted(notebook_root.glob("*.ipynb")))
    return notebook_paths


def test_all_analysis_notebooks_are_valid_json_and_code_cells_compile() -> None:
    # Проверяем только активный слой.
    # Архивные notebook живут отдельно и сюда не входят.
    notebook_paths = _iter_active_notebook_paths()

    assert notebook_paths

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        assert cells, f"Notebook has no cells: {notebook_path}"

        for cell_index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{notebook_path.name}_cell_{cell_index}", "exec")
