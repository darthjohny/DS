"""Журнал прогонов программы.

Назначение файла:
    - хранить логику создания markdown-файлов для прогонов пайплайна;
    - не смешивать журнал запусков со скриптами моделей и оркестратора;
    - давать отдельную и понятную точку входа для фиксации каждого run.

Что делает:
    1. Создаёт каталог `experiments/Логи работы программы`.
    2. Находит следующий номер файла по шаблону `run_XXX.md`.
    3. Создаёт новый markdown-файл с шаблоном:
       - источник данных;
       - режим запуска;
       - версии моделей;
       - размеры веток пайплайна;
       - итоговые наблюдения по run.

Что не делает:
    - не запускает пайплайн;
    - не пишет в БД;
    - не читает результаты автоматически;
    - не меняет параметры калибровки decision layer.

Использование:
    python src/program_run_logbook.py

После запуска будет создан следующий файл:
    experiments/Логи работы программы/run_XXX.md
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGBOOK_DIR = ROOT_DIR / "experiments" / "Логи работы программы"
RUN_PATTERN = re.compile(r"^run_(\d{3})\.md$")


def ensure_logbook_dir() -> Path:
    """Создаёт каталог журнала прогонов, если его ещё нет."""
    LOGBOOK_DIR.mkdir(parents=True, exist_ok=True)
    return LOGBOOK_DIR


def list_run_files(logbook_dir: Path) -> list[Path]:
    """Возвращает только валидные markdown-файлы прогонов."""
    files: list[Path] = []
    for path in sorted(logbook_dir.iterdir()):
        if path.is_file() and RUN_PATTERN.match(path.name):
            files.append(path)
    return files


def next_run_number(logbook_dir: Path) -> int:
    """Определяет следующий номер прогона по уже существующим файлам."""
    max_number = 0
    for path in list_run_files(logbook_dir):
        match = RUN_PATTERN.match(path.name)
        if match is None:
            continue
        max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def build_run_template(run_number: int) -> str:
    """Собирает шаблон markdown-файла для нового прогона."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"run_{run_number:03d}"
    return f"""# {run_id}

Дата: {created_at}
Статус: черновик

## Что запускали
- Скрипт:
- Источник входных данных:
- persist:
- limit:

## Версии моделей
- router_model_version:
- host_model_version:

## Параметры decision layer
- class_prior:
- metallicity_factor:
- distance_factor:
- quality_factor:

## Размеры потоков
- входных объектов:
- M/K/G/F dwarf:
- A/B/O:
- evolved:

## Сводка по score
- router_similarity:
- similarity:
- final_score:

## Top-N summary
- 

## Предупреждения
- 

## Ошибки
- 

## Вывод
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
