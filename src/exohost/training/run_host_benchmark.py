# Файл `run_host_benchmark.py` слоя `training`.
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

from exohost.datasets.build_host_field_training_dataset import (
    build_host_field_training_dataset,
)
from exohost.datasets.load_host_training_dataset import load_host_training_dataset
from exohost.datasets.load_router_training_dataset import load_router_training_dataset
from exohost.db.engine import make_read_only_engine
from exohost.evaluation.protocol import (
    HOST_FIELD_CLASSIFICATION_TASK,
    ClassificationTask,
)
from exohost.features.training_frame import (
    prepare_host_training_frame,
    prepare_router_training_frame,
)
from exohost.models.registry import build_router_model_specs, select_model_specs
from exohost.training.benchmark_orchestration import (
    build_task_ready_frame,
    raise_small_sample_error,
)
from exohost.training.benchmark_runner import BenchmarkRunResult, run_benchmark

HOST_BENCHMARK_TASKS: tuple[ClassificationTask, ...] = (HOST_FIELD_CLASSIFICATION_TASK,)
HOST_TASK_BY_NAME: dict[str, ClassificationTask] = {
    task.name: task for task in HOST_BENCHMARK_TASKS
}


def get_host_benchmark_task(task_name: str) -> ClassificationTask:
    # Возвращаем каноническую host benchmark-задачу по имени.
    try:
        return HOST_TASK_BY_NAME[task_name]
    except KeyError as error:
        supported_tasks = ", ".join(sorted(HOST_TASK_BY_NAME))
        raise ValueError(
            f"Unsupported host benchmark task: {task_name}. "
            f"Supported tasks: {supported_tasks}"
        ) from error


def run_host_benchmark_with_engine(
    engine: Engine,
    *,
    task_name: str,
    host_limit: int | None = None,
    router_limit: int | None = None,
    field_to_host_ratio: int = 1,
    selected_model_names: tuple[str, ...] | None = None,
) -> BenchmarkRunResult:
    # Выполняем host-vs-field benchmark на уже созданном read-only engine.
    task = get_host_benchmark_task(task_name)
    raw_host_frame = load_host_training_dataset(engine, limit=host_limit)
    raw_router_frame = load_router_training_dataset(engine, limit=router_limit)
    host_frame = prepare_host_training_frame(raw_host_frame)
    router_frame = prepare_router_training_frame(raw_router_frame)
    host_field_frame = build_host_field_training_dataset(
        host_frame,
        router_frame,
        field_to_host_ratio=field_to_host_ratio,
    )
    task_frame = build_task_ready_frame(
        host_field_frame,
        target_column=task.target_column,
        frame_name="host benchmark frame",
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
        if host_limit is not None or router_limit is not None:
            raise_small_sample_error(
                sample_name="Host benchmark",
                limit_hint="Increase --host-limit/--router-limit or run without limits.",
                cause=error,
            )
        raise


def run_host_benchmark_from_env(
    *,
    task_name: str,
    host_limit: int | None = None,
    router_limit: int | None = None,
    field_to_host_ratio: int = 1,
    selected_model_names: tuple[str, ...] | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> BenchmarkRunResult:
    # Создаем read-only engine из окружения и выполняем host benchmark.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        return run_host_benchmark_with_engine(
            engine,
            task_name=task_name,
            host_limit=host_limit,
            router_limit=router_limit,
            field_to_host_ratio=field_to_host_ratio,
            selected_model_names=selected_model_names,
        )
    finally:
        engine.dispose()
