# Файл `command.py` слоя `cli`.
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

from exohost.cli.task_catalog import PUBLIC_BENCHMARK_TASK_NAMES
from exohost.evaluation.hierarchical_tasks import HIERARCHICAL_TASK_BY_NAME
from exohost.evaluation.refinement_family_tasks import REFINEMENT_FAMILY_TASK_BY_NAME
from exohost.models.registry import SUPPORTED_MODEL_NAMES
from exohost.reporting.model_artifacts import (
    DEFAULT_MODEL_OUTPUT_DIR,
    save_model_artifacts,
)
from exohost.training.run_hierarchical_training import run_hierarchical_training_from_env
from exohost.training.run_host_benchmark import HOST_TASK_BY_NAME
from exohost.training.run_host_training import run_host_training_from_env
from exohost.training.run_refinement_family_training import (
    run_refinement_family_training_from_env,
)
from exohost.training.run_router_benchmark import ROUTER_TASK_BY_NAME
from exohost.training.run_router_training import run_router_training_from_env

from .support import (
    build_train_context,
    print_model_artifact_paths,
    print_train_stage,
    resolve_model_output_dir,
)


def register_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем train-команду первой волны для сохранения model artifacts.
    train_parser = subparsers.add_parser(
        "train",
        help="Обучение моделей V2.",
    )
    train_parser.add_argument(
        "--task",
        choices=PUBLIC_BENCHMARK_TASK_NAMES,
        default=None,
        help="Задача обучения для выбранного data source.",
    )
    train_parser.add_argument(
        "--model",
        dest="model_name",
        choices=SUPPORTED_MODEL_NAMES,
        default=None,
        help="Имя одной обучаемой модели.",
    )
    train_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный лимит строк из router training source.",
    )
    train_parser.add_argument(
        "--host-limit",
        type=int,
        default=None,
        help="Необязательный лимит строк из host training source.",
    )
    train_parser.add_argument(
        "--router-limit",
        type=int,
        default=None,
        help="Необязательный лимит строк из router training source для host-vs-field.",
    )
    train_parser.add_argument(
        "--field-to-host-ratio",
        type=int,
        default=1,
        help="Сколько field-объектов брать на одну host-звезду.",
    )
    train_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    train_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    train_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MODEL_OUTPUT_DIR),
        help="Каталог для сохранения model artifacts.",
    )
    train_parser.set_defaults(handler=handle_train)
def handle_train(namespace: argparse.Namespace) -> int:
    # Выполняем train-команду и сохраняем model artifacts.
    if namespace.task is None or namespace.model_name is None:
        return 0

    print_train_stage(f"start task={namespace.task} model={namespace.model_name}")
    if namespace.task in ROUTER_TASK_BY_NAME:
        train_result = run_router_training_from_env(
            task_name=namespace.task,
            model_name=namespace.model_name,
            limit=namespace.limit,
            dotenv_path=namespace.dotenv_path,
            connect_timeout=namespace.connect_timeout,
        )
    elif namespace.task in HOST_TASK_BY_NAME:
        train_result = run_host_training_from_env(
            task_name=namespace.task,
            model_name=namespace.model_name,
            host_limit=namespace.host_limit,
            router_limit=namespace.router_limit,
            field_to_host_ratio=namespace.field_to_host_ratio,
            dotenv_path=namespace.dotenv_path,
            connect_timeout=namespace.connect_timeout,
        )
    elif namespace.task in HIERARCHICAL_TASK_BY_NAME:
        train_result = run_hierarchical_training_from_env(
            task_name=namespace.task,
            model_name=namespace.model_name,
            limit=namespace.limit,
            dotenv_path=namespace.dotenv_path,
            connect_timeout=namespace.connect_timeout,
        )
    elif namespace.task in REFINEMENT_FAMILY_TASK_BY_NAME:
        train_result = run_refinement_family_training_from_env(
            task_name=namespace.task,
            model_name=namespace.model_name,
            limit=namespace.limit,
            dotenv_path=namespace.dotenv_path,
            connect_timeout=namespace.connect_timeout,
        )
    else:
        raise RuntimeError(f"Unsupported train task dispatch: {namespace.task}")

    print_train_stage("save artifacts")
    model_artifact_paths = save_model_artifacts(
        train_result,
        output_dir=resolve_model_output_dir(namespace),
        extra_metadata=build_train_context(namespace),
    )
    print_model_artifact_paths(model_artifact_paths)
    return 0
