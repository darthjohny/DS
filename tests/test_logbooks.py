"""Точечные тесты markdown-logbook helper-ов и CLI-журналов."""

from __future__ import annotations

import re
from pathlib import Path

from infra.logbook import ensure_logbook_dir, list_markdown_files, next_markdown_number
from logbooks.decision_layer import create_iteration_file
from logbooks.program_run import create_run_file


def test_ensure_logbook_dir_creates_missing_parents(tmp_path: Path) -> None:
    """Helper должен создавать каталог журнала вместе с родителями."""
    logbook_dir = tmp_path / "nested" / "logs"

    created = ensure_logbook_dir(logbook_dir)

    assert created == logbook_dir
    assert logbook_dir.exists()
    assert logbook_dir.is_dir()


def test_list_markdown_files_filters_and_sorts_only_matching_names(tmp_path: Path) -> None:
    """Helper должен возвращать только валидные markdown-файлы в стабильном порядке."""
    pattern = re.compile(r"^run_(\d{3})\.md$")
    (tmp_path / "run_010.md").write_text("", encoding="utf-8")
    (tmp_path / "run_002.md").write_text("", encoding="utf-8")
    (tmp_path / "notes.md").write_text("", encoding="utf-8")
    (tmp_path / "run_999.txt").write_text("", encoding="utf-8")

    files = list_markdown_files(tmp_path, pattern)

    assert [path.name for path in files] == ["run_002.md", "run_010.md"]


def test_next_markdown_number_ignores_non_matching_files(tmp_path: Path) -> None:
    """Следующий номер должен считаться по максимальному валидному файлу."""
    pattern = re.compile(r"^iteration_(\d{3})\.md$")
    (tmp_path / "iteration_002.md").write_text("", encoding="utf-8")
    (tmp_path / "iteration_007.md").write_text("", encoding="utf-8")
    (tmp_path / "iteration_old.md").write_text("", encoding="utf-8")

    assert next_markdown_number(tmp_path, pattern) == 8


def test_create_run_file_uses_next_number_and_writes_template(tmp_path: Path) -> None:
    """Журнал прогонов должен создавать следующий numbered markdown-файл."""
    (tmp_path / "run_001.md").write_text("# run_001\n", encoding="utf-8")
    (tmp_path / "run_003.md").write_text("# run_003\n", encoding="utf-8")

    created_path = create_run_file(tmp_path)
    content = created_path.read_text(encoding="utf-8")

    assert created_path.name == "run_004.md"
    assert content.startswith("# run_004")
    assert "## Контекст запуска" in content
    assert "## Следующий шаг" in content


def test_create_iteration_file_uses_next_number_and_writes_template(tmp_path: Path) -> None:
    """Журнал калибровки должен создавать следующий iteration markdown-файл."""
    (tmp_path / "iteration_004.md").write_text("# iteration_004\n", encoding="utf-8")

    created_path = create_iteration_file(tmp_path)
    content = created_path.read_text(encoding="utf-8")

    assert created_path.name == "iteration_005.md"
    assert content.startswith("# iteration_005")
    assert "## Что меняем" in content
    assert "## Следующий шаг" in content
