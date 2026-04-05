# Тестовый файл `test_run_router_benchmark.py` домена `training`.
#
# Этот файл проверяет только:
# - проверку логики домена: обучающие orchestration-сценарии и benchmark-runner;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `training` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd
import pytest

from exohost.training.benchmark_orchestration import build_task_ready_frame
from exohost.training.run_router_benchmark import (
    get_router_benchmark_task,
)


def test_get_router_benchmark_task_returns_known_task() -> None:
    # Проверяем, что реестр задач возвращает каноническую задачу по имени.
    task = get_router_benchmark_task("spectral_class_classification")

    assert task.target_column == "spec_class"


def test_get_router_benchmark_task_rejects_unknown_name() -> None:
    # Неизвестные router benchmark-задачи должны отбрасываться явно.
    with pytest.raises(ValueError, match="Unsupported router benchmark task"):
        get_router_benchmark_task("unknown_task")


def test_build_task_ready_frame_rejects_missing_target_labels() -> None:
    # Если в источнике нет usable target-меток, benchmark должен падать явно.
    frame = pd.DataFrame(
        {
            "source_id": [1, 2],
            "spec_class": ["G", "K"],
            "spec_subclass": [pd.NA, pd.NA],
            "evolution_stage": ["dwarf", "evolved"],
        }
    )

    with pytest.raises(RuntimeError, match="does not contain usable target labels"):
        build_task_ready_frame(
            frame,
            target_column="spec_subclass",
            frame_name="router training frame",
        )
