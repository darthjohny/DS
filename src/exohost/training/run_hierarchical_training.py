# Файл `run_hierarchical_training.py` слоя `training`.
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
from exohost.models.registry import build_router_model_specs, get_model_spec
from exohost.training.benchmark_orchestration import build_task_ready_frame
from exohost.training.hierarchical_source import load_hierarchical_prepared_training_frame
from exohost.training.run_hierarchical_benchmark import get_hierarchical_benchmark_task
from exohost.training.train_runner import TrainRunResult, run_training


def run_hierarchical_training_with_engine(
    engine: Engine,
    *,
    task_name: str,
    model_name: str,
    limit: int | None = None,
) -> TrainRunResult:
    # Выполняем training по одной hierarchical-задаче.
    task = get_hierarchical_benchmark_task(task_name)
    prepared_frame = load_hierarchical_prepared_training_frame(
        engine,
        task_name=task_name,
        limit=limit,
    )
    task_frame = build_task_ready_frame(
        prepared_frame,
        target_column=task.target_column,
        frame_name="hierarchical training frame",
    )
    model_spec = get_model_spec(
        build_router_model_specs(task.feature_columns),
        model_name=model_name,
    )
    return run_training(
        task_frame,
        task=task,
        model_spec=model_spec,
    )


def run_hierarchical_training_from_env(
    *,
    task_name: str,
    model_name: str,
    limit: int | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> TrainRunResult:
    # Создаем read-only engine и выполняем hierarchical training.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        return run_hierarchical_training_with_engine(
            engine,
            task_name=task_name,
            model_name=model_name,
            limit=limit,
        )
    finally:
        engine.dispose()
