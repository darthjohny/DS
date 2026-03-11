"""Общая инфраструктура для markdown-журналов проекта.

Модуль инкапсулирует файловые операции, общие для разных logbook-сценариев:

- создание каталога журнала;
- фильтрацию валидных markdown-файлов по шаблону имени;
- вычисление следующего числового суффикса.
"""

from __future__ import annotations

from pathlib import Path
from re import Pattern


def ensure_logbook_dir(logbook_dir: Path) -> Path:
    """Создать каталог журнала и вернуть его путь.

    Побочные эффекты
    ----------------
    Создаёт каталог на диске вместе с недостающими родительскими
    директориями.
    """
    logbook_dir.mkdir(parents=True, exist_ok=True)
    return logbook_dir


def list_markdown_files(
    logbook_dir: Path,
    pattern: Pattern[str],
) -> list[Path]:
    """Вернуть markdown-файлы журнала, подходящие под шаблон нумерации.

    Функция читает содержимое каталога и отбрасывает посторонние файлы,
    оставляя только те, чьи имена совпадают с ожидаемым regex-шаблоном.
    """
    files: list[Path] = []
    for path in sorted(logbook_dir.iterdir()):
        if path.is_file() and pattern.match(path.name):
            files.append(path)
    return files


def next_markdown_number(
    logbook_dir: Path,
    pattern: Pattern[str],
) -> int:
    """Вычислить следующий числовой суффикс для нового markdown-файла.

    Нумерация строится от максимального уже существующего номера среди
    валидных файлов журнала.
    """
    max_number = 0
    for path in list_markdown_files(logbook_dir, pattern):
        match = pattern.match(path.name)
        if match is None:
            continue
        max_number = max(max_number, int(match.group(1)))
    return max_number + 1
