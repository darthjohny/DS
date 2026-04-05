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

from exohost.reporting.model_artifacts import ModelArtifactPaths


def print_train_stage(message: str) -> None:
    # Печатаем короткий статус train-команды.
    print(f"[train] {message}")


def print_model_artifact_paths(paths: ModelArtifactPaths) -> None:
    # Печатаем каталог сохраненных model artifacts.
    print(f"[artifacts] saved_to={paths.run_dir}")


def build_train_context(namespace: argparse.Namespace) -> dict[str, object]:
    # Собираем metadata-контекст train-команды.
    context: dict[str, object] = {
        "task": None if namespace.task is None else str(namespace.task),
        "model_name": None if namespace.model_name is None else str(namespace.model_name),
        "dotenv_path": str(namespace.dotenv_path),
        "connect_timeout": int(namespace.connect_timeout),
        "output_dir": str(namespace.output_dir),
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


def resolve_model_output_dir(namespace: argparse.Namespace) -> Path:
    # Приводим output dir train-команды к Path.
    return Path(namespace.output_dir)
