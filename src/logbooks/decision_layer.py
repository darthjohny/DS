"""Журнал калибровки decision layer.

Назначение файла:
    - хранить логику создания markdown-файлов для ручной калибровки
      итогового `final_score`;
    - не смешивать журнал экспериментов с `star_orchestrator.py`;
    - давать отдельную, видимую точку входа для фиксации каждой итерации.

Что делает:
    1. Создаёт каталог `experiments/Логи калибровки decision_layer`.
    2. Находит следующий номер итерации по шаблону `iteration_XXX.md`.
    3. Создаёт новый markdown-файл с готовым шаблоном:
       - какие параметры меняли;
       - какие ожидания были;
       - что получилось;
       - какой следующий шаг.

Что не делает:
    - не меняет пайплайн;
    - не пишет в БД;
    - не запускает модели;
    - не анализирует результаты автоматически.
"""

from __future__ import annotations

import re
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from infra.logbook import (
    ensure_logbook_dir as _ensure_logbook_dir,
)
from infra.logbook import (
    list_markdown_files,
    next_markdown_number,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
LOGBOOK_DIR = ROOT_DIR / "experiments" / "Логи калибровки decision_layer"
ITERATION_PATTERN = re.compile(r"^iteration_(\d{3})\.md$")


def ensure_logbook_dir() -> Path:
    """Создаёт каталог журнала, если его ещё нет."""
    return _ensure_logbook_dir(LOGBOOK_DIR)


def list_iteration_files(logbook_dir: Path) -> list[Path]:
    """Возвращает только валидные markdown-файлы итераций."""
    return list_markdown_files(logbook_dir, ITERATION_PATTERN)


def next_iteration_number(logbook_dir: Path) -> int:
    """Определяет следующий номер итерации по существующим файлам."""
    return next_markdown_number(logbook_dir, ITERATION_PATTERN)


def build_iteration_template(iteration_number: int) -> str:
    """Собирает шаблон markdown-файла для новой калибровки."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    iteration_id = f"iteration_{iteration_number:03d}"
    return f"""# {iteration_id}

Дата: {created_at}
Статус: черновик

## Контекст итерации
- relation:
- source_name:
- run_id:
- top_n:

## Что меняем
- class_prior:
- metallicity_factor:
- distance_factor:
- quality_factor:
- host_score semantics:

## Формула
`final_score = host_posterior × class_prior × distance_factor × quality_factor × metallicity_factor`

## Режимы скоринга
- router_score_mode:
- host_score_mode:
- host_model_version:

## Гипотеза
-

## Параметры итерации
### class_prior
- K:
- G:
- M:
- F:

### metallicity_factor
-

### distance_factor
-

### quality_factor
- ruwe:
- parallax_over_error:

## Ожидаемый эффект
-

## Фактический результат
-

## Сводка по top-N
-

## Итог
-

## Следующий шаг
-
"""


def create_iteration_file(logbook_dir: Path) -> Path:
    """Создаёт следующий файл итерации и возвращает путь к нему."""
    iteration_number = next_iteration_number(logbook_dir)
    path = logbook_dir / f"iteration_{iteration_number:03d}.md"
    path.write_text(
        build_iteration_template(iteration_number),
        encoding="utf-8",
    )
    return path


def parse_args() -> Namespace:
    """Разбирает аргументы CLI."""
    parser = ArgumentParser(
        description="Создаёт новый markdown-файл для калибровки decision layer."
    )
    return parser.parse_args()


def main() -> None:
    """Создаёт следующий файл журнала калибровки."""
    parse_args()
    logbook_dir = ensure_logbook_dir()
    created_path = create_iteration_file(logbook_dir)
    print(f"Создан файл калибровки: {created_path}")


if __name__ == "__main__":
    main()
