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
from exohost.reporting.benchmark_artifacts import (
    DEFAULT_BENCHMARK_OUTPUT_DIR,
    save_benchmark_artifacts,
)
from exohost.training.run_hierarchical_benchmark import (
    run_hierarchical_benchmark_from_env,
)
from exohost.training.run_host_benchmark import HOST_TASK_BY_NAME, run_host_benchmark_from_env
from exohost.training.run_refinement_family_benchmark import (
    run_refinement_family_benchmark_from_env,
)
from exohost.training.run_router_benchmark import ROUTER_TASK_BY_NAME, run_router_benchmark_from_env

from .support import (
    build_benchmark_context,
    format_benchmark_tables,
    parse_model_names,
    print_benchmark_artifact_paths,
    print_benchmark_result,
    print_benchmark_stage,
    resolve_benchmark_output_dir,
)


def register_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Регистрируем benchmark-команду первой волны.
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Запуск benchmark-контура V2.",
    )
    benchmark_parser.add_argument(
        "--task",
        choices=PUBLIC_BENCHMARK_TASK_NAMES,
        default="spectral_class_classification",
        help="Задача benchmark-контура.",
    )
    benchmark_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный лимит строк из router training source.",
    )
    benchmark_parser.add_argument(
        "--host-limit",
        type=int,
        default=None,
        help="Необязательный лимит строк из host training source.",
    )
    benchmark_parser.add_argument(
        "--router-limit",
        type=int,
        default=None,
        help="Необязательный лимит строк из router training source для host-vs-field.",
    )
    benchmark_parser.add_argument(
        "--field-to-host-ratio",
        type=int,
        default=1,
        help="Сколько field-объектов брать на одну host-звезду.",
    )
    benchmark_parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Путь к .env файлу с параметрами подключения.",
    )
    benchmark_parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Таймаут подключения к БД в секундах.",
    )
    benchmark_parser.add_argument(
        "--models",
        default=None,
        help=f"Список моделей через запятую. Поддерживаемые: {', '.join(SUPPORTED_MODEL_NAMES)}",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_BENCHMARK_OUTPUT_DIR),
        help="Каталог для сохранения benchmark-артефактов.",
    )
    benchmark_parser.set_defaults(handler=handle_benchmark)
def handle_benchmark(namespace: argparse.Namespace) -> int:
    # Выполняем benchmark-команду и сохраняем artifacts.
    selected_model_names = parse_model_names(namespace.models)
    print_benchmark_stage(f"start task={namespace.task}")
    if namespace.task in ROUTER_TASK_BY_NAME:
        benchmark_result = run_router_benchmark_from_env(
            task_name=namespace.task,
            limit=namespace.limit,
            selected_model_names=selected_model_names,
            dotenv_path=namespace.dotenv_path,
            connect_timeout=namespace.connect_timeout,
        )
    elif namespace.task in HOST_TASK_BY_NAME:
        benchmark_result = run_host_benchmark_from_env(
            task_name=namespace.task,
            host_limit=namespace.host_limit,
            router_limit=namespace.router_limit,
            field_to_host_ratio=namespace.field_to_host_ratio,
            selected_model_names=selected_model_names,
            dotenv_path=namespace.dotenv_path,
            connect_timeout=namespace.connect_timeout,
        )
    elif namespace.task in HIERARCHICAL_TASK_BY_NAME:
        benchmark_result = run_hierarchical_benchmark_from_env(
            task_name=namespace.task,
            limit=namespace.limit,
            selected_model_names=selected_model_names,
            dotenv_path=namespace.dotenv_path,
            connect_timeout=namespace.connect_timeout,
        )
    elif namespace.task in REFINEMENT_FAMILY_TASK_BY_NAME:
        benchmark_result = run_refinement_family_benchmark_from_env(
            task_name=namespace.task,
            limit=namespace.limit,
            selected_model_names=selected_model_names,
            dotenv_path=namespace.dotenv_path,
            connect_timeout=namespace.connect_timeout,
        )
    else:
        raise RuntimeError(f"Unsupported benchmark task dispatch: {namespace.task}")

    print_benchmark_stage("save artifacts")
    metrics_text = benchmark_result.metrics_df.to_string(index=False)
    cv_text = benchmark_result.cv_summary_df.to_string(index=False)
    benchmark_artifact_paths = save_benchmark_artifacts(
        benchmark_result,
        output_dir=resolve_benchmark_output_dir(namespace),
        extra_metadata=build_benchmark_context(namespace),
    )
    print_benchmark_result(
        benchmark_result.task_name,
        format_benchmark_tables(metrics_text, cv_text),
    )
    print_benchmark_artifact_paths(benchmark_artifact_paths)
    return 0
