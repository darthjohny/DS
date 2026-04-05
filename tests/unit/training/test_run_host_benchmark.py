# Тестовый файл `test_run_host_benchmark.py` домена `training`.
#
# Этот файл проверяет только:
# - проверку логики домена: обучающие orchestration-сценарии и benchmark-runner;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `training` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pytest

from exohost.models.registry import build_router_model_specs, select_model_specs
from exohost.training.run_host_benchmark import get_host_benchmark_task


def test_get_host_benchmark_task_returns_known_task() -> None:
    # Проверяем, что host benchmark-реестр возвращает каноническую задачу.
    task = get_host_benchmark_task("host_field_classification")

    assert task.target_column == "host_label"


def test_get_host_benchmark_task_rejects_unknown_name() -> None:
    # Неизвестные host benchmark-задачи должны отбрасываться явно.
    with pytest.raises(ValueError, match="Unsupported host benchmark task"):
        get_host_benchmark_task("unknown_host_task")


def test_select_model_specs_returns_requested_order() -> None:
    # Поднабор моделей должен возвращаться в запрошенном порядке.
    model_specs = build_router_model_specs(("teff_gspphot",))

    selected_specs = select_model_specs(
        model_specs,
        selected_model_names=("mlp_classifier", "gmm_classifier"),
    )

    assert [model_spec.model_name for model_spec in selected_specs] == [
        "mlp_classifier",
        "gmm_classifier",
    ]
