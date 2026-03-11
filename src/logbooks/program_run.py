"""Журнал прогонов программы.

Назначение файла:
    - хранить логику создания markdown-файлов для прогонов пайплайна;
    - не смешивать журнал запусков со скриптами моделей и оркестратора;
    - давать отдельную и понятную точку входа для фиксации каждого прогона.

Что делает:
    1. Создаёт каталог `experiments/Логи работы программы`.
    2. Находит следующий номер файла по шаблону `run_XXX.md`.
    3. Создаёт новый markdown-файл с шаблоном:
       - источник данных;
       - режим запуска;
       - версии моделей;
       - размеры веток пайплайна;
       - итоговые наблюдения по прогону.

Что не делает:
    - не запускает пайплайн;
    - не пишет в БД;
    - не читает результаты автоматически;
    - не меняет параметры калибровки decision layer.
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
LOGBOOK_DIR = ROOT_DIR / "experiments" / "Логи работы программы"
RUN_PATTERN = re.compile(r"^run_(\d{3})\.md$")


def ensure_logbook_dir() -> Path:
    """Создаёт каталог журнала прогонов, если его ещё нет."""
    return _ensure_logbook_dir(LOGBOOK_DIR)


def list_run_files(logbook_dir: Path) -> list[Path]:
    """Возвращает только валидные markdown-файлы прогонов."""
    return list_markdown_files(logbook_dir, RUN_PATTERN)


def next_run_number(logbook_dir: Path) -> int:
    """Определяет следующий номер прогона по уже существующим файлам."""
    return next_markdown_number(logbook_dir, RUN_PATTERN)


def build_run_template(run_number: int) -> str:
    """Собирает шаблон markdown-файла для нового прогона."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"run_{run_number:03d}"
    return f"""# {run_id}

Дата: {created_at}
Статус: черновик

## Контекст запуска
- Скрипт:
- Команда:
- Источник входных данных:
- limit:
- persist:

## Артефакты и режимы
- router_model_version:
- host_model_version:
- router_score_mode:
- host_score_mode:

## Параметры decision layer
- Формула:
- class_prior:
- quality_factor:
- metallicity_factor:
- color_factor:
- validation_factor:

## Размеры потоков
- input_rows:
- router_rows:
- host_rows:
- low_rows:
- priority_rows:

## Ключевые метрики
- router_log_posterior:
- host_posterior:
- host_log_lr:
- final_score:
- распределение `priority_tier`:

## Кандидаты top-N
-

## Артефакты
- run_id:
- router_results_table:
- priority_results_table:

## Предупреждения
-

## Ошибки
-

## Итог
-

## Следующий шаг
-
"""


def create_run_file(logbook_dir: Path) -> Path:
    """Создаёт следующий markdown-файл прогона и возвращает путь к нему."""
    run_number = next_run_number(logbook_dir)
    path = logbook_dir / f"run_{run_number:03d}.md"
    path.write_text(build_run_template(run_number), encoding="utf-8")
    return path


def parse_args() -> Namespace:
    """Разбирает аргументы CLI."""
    parser = ArgumentParser(
        description="Создаёт новый markdown-файл для журнала прогонов."
    )
    return parser.parse_args()


def main() -> None:
    """Создаёт следующий файл журнала прогонов."""
    parse_args()
    logbook_dir = ensure_logbook_dir()
    created_path = create_run_file(logbook_dir)
    print(f"Создан файл журнала прогона: {created_path}")


if __name__ == "__main__":
    main()
