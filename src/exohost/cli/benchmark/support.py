# Файл `support.py` слоя `cli`.
#
# Этот файл отвечает только за:
# - CLI-команды и orchestration entrypoints;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - CLI-команды или support-модули этого же домена;
# - пользовательский запуск через `python -m exohost.cli.main`.

from __future__ import annotations

import argparse
from pathlib import Path

from exohost.models.registry import SUPPORTED_MODEL_NAMES
from exohost.reporting.benchmark_artifacts import BenchmarkArtifactPaths


def print_benchmark_stage(message: str) -> None:
    # Печатаем короткий статус benchmark-оркестратора.
    print(f"[benchmark] {message}")


def print_benchmark_result(task_name: str, result_text: str) -> None:
    # Печатаем компактный итог benchmark-команды в консоль.
    print(f"[benchmark] task={task_name}")
    print(result_text)


def print_benchmark_artifact_paths(paths: BenchmarkArtifactPaths) -> None:
    # Печатаем каталог сохраненных benchmark-артефактов.
    print(f"[artifacts] saved_to={paths.run_dir}")


def format_benchmark_tables(metrics_text: str, cv_text: str) -> str:
    # Собираем табличный вывод benchmark-команды в один консольный блок.
    return "\n\n".join(
        (
            "=== metrics ===",
            metrics_text,
            "=== cv_summary ===",
            cv_text,
        )
    )


def parse_model_names(raw_value: str | None) -> tuple[str, ...] | None:
    # Преобразуем CLI-строку в канонический список имен моделей.
    if raw_value is None:
        return None

    selected_names = tuple(
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    )
    if not selected_names:
        raise ValueError(
            "Benchmark model list is empty. "
            f"Supported models: {', '.join(SUPPORTED_MODEL_NAMES)}"
        )
    return selected_names


def build_benchmark_context(namespace: argparse.Namespace) -> dict[str, object]:
    # Собираем metadata-контекст benchmark-команды.
    context: dict[str, object] = {
        "task": str(namespace.task),
        "dotenv_path": str(namespace.dotenv_path),
        "connect_timeout": int(namespace.connect_timeout),
        "output_dir": str(namespace.output_dir),
        "selected_models": parse_model_names(namespace.models),
    }
    for field_name in (
        "limit",
        "host_limit",
        "router_limit",
        "field_to_host_ratio",
    ):
        field_value = getattr(namespace, field_name, None)
        if field_value is not None:
            context[field_name] = field_value
    return context


def resolve_benchmark_output_dir(namespace: argparse.Namespace) -> Path:
    # Приводим output dir benchmark-команды к Path.
    return Path(namespace.output_dir)
