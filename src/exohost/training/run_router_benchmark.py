# Файл `run_router_benchmark.py` слоя `training`.
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

from exohost.datasets.load_router_training_dataset import load_router_training_dataset
from exohost.db.engine import make_read_only_engine
from exohost.evaluation.protocol import (
    SPECTRAL_CLASS_CLASSIFICATION_TASK,
    SPECTRAL_SUBCLASS_CLASSIFICATION_TASK,
    STAGE_CLASSIFICATION_TASK,
    ClassificationTask,
)
from exohost.features.training_frame import prepare_router_training_frame
from exohost.models.registry import build_router_model_specs, select_model_specs
from exohost.training.benchmark_orchestration import (
    build_task_ready_frame,
    raise_small_sample_error,
)
from exohost.training.benchmark_runner import BenchmarkRunResult, run_benchmark

ROUTER_BENCHMARK_TASKS: tuple[ClassificationTask, ...] = (
    STAGE_CLASSIFICATION_TASK,
    SPECTRAL_CLASS_CLASSIFICATION_TASK,
    SPECTRAL_SUBCLASS_CLASSIFICATION_TASK,
)

ROUTER_TASK_BY_NAME: dict[str, ClassificationTask] = {
    task.name: task for task in ROUTER_BENCHMARK_TASKS
}


def get_router_benchmark_task(task_name: str) -> ClassificationTask:
    # Возвращаем каноническую router-задачу по имени CLI/benchmark-контура.
    try:
        return ROUTER_TASK_BY_NAME[task_name]
    except KeyError as error:
        supported_tasks = ", ".join(sorted(ROUTER_TASK_BY_NAME))
        raise ValueError(
            f"Unsupported router benchmark task: {task_name}. "
            f"Supported tasks: {supported_tasks}"
        ) from error


def run_router_benchmark_with_engine(
    engine: Engine,
    *,
    task_name: str,
    limit: int | None = None,
    selected_model_names: tuple[str, ...] | None = None,
) -> BenchmarkRunResult:
    # Выполняем полный router benchmark на уже созданном read-only engine.
    task = get_router_benchmark_task(task_name)
    raw_frame = load_router_training_dataset(engine, limit=limit)
    prepared_frame = prepare_router_training_frame(raw_frame)
    task_frame = build_task_ready_frame(
        prepared_frame,
        target_column=task.target_column,
        frame_name="router training frame",
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
                sample_name="Router benchmark",
                limit_hint="Increase --limit or run without --limit.",
                cause=error,
            )
        raise


def run_router_benchmark_from_env(
    *,
    task_name: str,
    limit: int | None = None,
    selected_model_names: tuple[str, ...] | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> BenchmarkRunResult:
    # Создаем read-only engine из окружения и выполняем benchmark-прогон.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        return run_router_benchmark_with_engine(
            engine,
            task_name=task_name,
            limit=limit,
            selected_model_names=selected_model_names,
        )
    finally:
        engine.dispose()
