# Файл `run_refinement_family_benchmark.py` слоя `training`.
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
from exohost.evaluation.protocol import ClassificationTask
from exohost.evaluation.refinement_family_tasks import REFINEMENT_FAMILY_TASK_BY_NAME
from exohost.models.registry import build_router_model_specs, select_model_specs
from exohost.training.benchmark_orchestration import (
    build_task_ready_frame,
    raise_small_sample_error,
)
from exohost.training.benchmark_runner import BenchmarkRunResult, run_benchmark
from exohost.training.refinement_family_source import (
    load_refinement_family_prepared_training_frame,
)


def get_refinement_family_benchmark_task(task_name: str) -> ClassificationTask:
    # Возвращаем каноническую second-wave refinement family task по имени.
    try:
        return REFINEMENT_FAMILY_TASK_BY_NAME[task_name].task
    except KeyError as error:
        supported_tasks = ", ".join(sorted(REFINEMENT_FAMILY_TASK_BY_NAME))
        raise ValueError(
            f"Unsupported refinement family benchmark task: {task_name}. "
            f"Supported tasks: {supported_tasks}"
        ) from error


def run_refinement_family_benchmark_with_engine(
    engine: Engine,
    *,
    task_name: str,
    limit: int | None = None,
    selected_model_names: tuple[str, ...] | None = None,
) -> BenchmarkRunResult:
    # Выполняем benchmark по одной second-wave refinement family task.
    task = get_refinement_family_benchmark_task(task_name)
    prepared_frame = load_refinement_family_prepared_training_frame(
        engine,
        task_name=task_name,
        limit=limit,
    )
    task_frame = build_task_ready_frame(
        prepared_frame,
        target_column=task.target_column,
        frame_name="refinement family benchmark frame",
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
                sample_name="Refinement family benchmark",
                limit_hint="Increase --limit or run without --limit.",
                cause=error,
            )
        raise


def run_refinement_family_benchmark_from_env(
    *,
    task_name: str,
    limit: int | None = None,
    selected_model_names: tuple[str, ...] | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> BenchmarkRunResult:
    # Создаем read-only engine и выполняем second-wave refinement family benchmark.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        return run_refinement_family_benchmark_with_engine(
            engine,
            task_name=task_name,
            limit=limit,
            selected_model_names=selected_model_names,
        )
    finally:
        engine.dispose()
