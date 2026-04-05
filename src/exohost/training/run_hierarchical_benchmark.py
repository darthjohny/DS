# Файл `run_hierarchical_benchmark.py` слоя `training`.
#
# Этот файл отвечает только за:
# - оркестрацию обучения и benchmark-прогонов;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `training` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from sqlalchemy.engine import Engine

from exohost.db.engine import make_read_only_engine
from exohost.evaluation.hierarchical_tasks import (
    HIERARCHICAL_TASK_BY_NAME,
)
from exohost.evaluation.protocol import ClassificationTask
from exohost.models.registry import build_router_model_specs, select_model_specs
from exohost.training.benchmark_orchestration import (
    build_task_ready_frame,
    raise_small_sample_error,
)
from exohost.training.benchmark_runner import BenchmarkRunResult, run_benchmark
from exohost.training.hierarchical_source import load_hierarchical_prepared_training_frame


def get_hierarchical_benchmark_task(task_name: str) -> ClassificationTask:
    # Возвращаем каноническую hierarchical-задачу по имени.
    try:
        return HIERARCHICAL_TASK_BY_NAME[task_name]
    except KeyError as error:
        supported_tasks = ", ".join(sorted(HIERARCHICAL_TASK_BY_NAME))
        raise ValueError(
            f"Unsupported hierarchical benchmark task: {task_name}. "
            f"Supported tasks: {supported_tasks}"
        ) from error


def run_hierarchical_benchmark_with_engine(
    engine: Engine,
    *,
    task_name: str,
    limit: int | None = None,
    selected_model_names: tuple[str, ...] | None = None,
) -> BenchmarkRunResult:
    # Выполняем benchmark по одной hierarchical-задаче.
    task = get_hierarchical_benchmark_task(task_name)
    prepared_frame = load_hierarchical_prepared_training_frame(
        engine,
        task_name=task_name,
        limit=limit,
    )
    task_frame = build_task_ready_frame(
        prepared_frame,
        target_column=task.target_column,
        frame_name="hierarchical benchmark frame",
    )
    model_specs = select_model_specs(
        build_router_model_specs(task.feature_columns),
        selected_model_names=selected_model_names,
    )
    try:
        return run_benchmark(
            task_frame,
            task=task,
            model_specs=model_specs,
        )
    except ValueError as error:
        if limit is not None:
            raise_small_sample_error(
                sample_name="Hierarchical benchmark",
                limit_hint="Increase --limit or run without --limit.",
                cause=error,
            )
        raise


def run_hierarchical_benchmark_from_env(
    *,
    task_name: str,
    limit: int | None = None,
    selected_model_names: tuple[str, ...] | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> BenchmarkRunResult:
    # Создаем read-only engine и выполняем hierarchical benchmark.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        return run_hierarchical_benchmark_with_engine(
            engine,
            task_name=task_name,
            limit=limit,
            selected_model_names=selected_model_names,
        )
    finally:
        engine.dispose()
