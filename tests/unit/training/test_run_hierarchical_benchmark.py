# Тестовый файл `test_run_hierarchical_benchmark.py` домена `training`.
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
from sqlalchemy.engine import Engine

from exohost.training.run_hierarchical_benchmark import (
    get_hierarchical_benchmark_task,
    run_hierarchical_benchmark_with_engine,
)


def build_ood_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(40):
        rows.append(
            {
                "source_id": index + 1,
                "domain_target": "id" if index < 20 else "ood",
                "teff_gspphot": 5500.0 + index,
                "logg_gspphot": 4.2,
                "mh_gspphot": 0.0,
                "bp_rp": 0.8,
                "parallax": 10.0,
                "parallax_over_error": 12.0,
                "ruwe": 1.0,
                "phot_g_mean_mag": 12.0,
            }
        )
    return pd.DataFrame(rows)


def test_get_hierarchical_benchmark_task_returns_known_task() -> None:
    task = get_hierarchical_benchmark_task("gaia_id_ood_classification")

    assert task.target_column == "domain_target"


def test_get_hierarchical_benchmark_task_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported hierarchical benchmark task"):
        get_hierarchical_benchmark_task("unknown_task")


def test_run_hierarchical_benchmark_with_engine_returns_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "exohost.training.run_hierarchical_benchmark.load_hierarchical_prepared_training_frame",
        lambda engine, task_name, limit=None: build_ood_frame(),
    )

    result = run_hierarchical_benchmark_with_engine(
        engine=Engine.__new__(Engine),
        task_name="gaia_id_ood_classification",
        limit=12,
        selected_model_names=("hist_gradient_boosting",),
    )

    assert result.task_name == "gaia_id_ood_classification"
    assert result.metrics_df["model_name"].tolist() == ["hist_gradient_boosting", "hist_gradient_boosting"]
