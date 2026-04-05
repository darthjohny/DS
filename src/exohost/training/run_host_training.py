# Файл `run_host_training.py` слоя `training`.
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
from exohost.features.training_frame import (
    prepare_host_training_frame,
    prepare_router_training_frame,
)
from exohost.models.registry import build_router_model_specs, get_model_spec
from exohost.training.benchmark_orchestration import build_task_ready_frame
from exohost.training.run_host_benchmark import get_host_benchmark_task
from exohost.training.train_runner import TrainRunResult, run_training


def run_host_training_with_engine(
    engine: Engine,
    *,
    task_name: str,
    model_name: str,
    host_limit: int | None = None,
    router_limit: int | None = None,
    field_to_host_ratio: int = 1,
) -> TrainRunResult:
    # Выполняем host-vs-field training на уже созданном read-only engine.
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
        frame_name="host training frame",
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


def run_host_training_from_env(
    *,
    task_name: str,
    model_name: str,
    host_limit: int | None = None,
    router_limit: int | None = None,
    field_to_host_ratio: int = 1,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> TrainRunResult:
    # Создаем read-only engine из окружения и выполняем host training.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        return run_host_training_with_engine(
            engine,
            task_name=task_name,
            model_name=model_name,
            host_limit=host_limit,
            router_limit=router_limit,
            field_to_host_ratio=field_to_host_ratio,
        )
    finally:
        engine.dispose()
