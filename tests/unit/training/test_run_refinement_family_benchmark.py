# Тестовый файл `test_run_refinement_family_benchmark.py` домена `training`.
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

from exohost.training.run_refinement_family_benchmark import (
    get_refinement_family_benchmark_task,
    run_refinement_family_benchmark_with_engine,
)


def build_family_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(40):
        rows.append(
            {
                "source_id": index + 1,
                "spectral_subclass": "2" if index < 20 else "4",
                "teff_gspphot": 5500.0 + index,
                "logg_gspphot": 4.2,
                "mh_gspphot": 0.0,
                "bp_rp": 0.8,
                "parallax": 10.0,
                "parallax_over_error": 12.0,
                "ruwe": 1.0,
                "radius_flame": 1.0,
                "lum_flame": 1.1,
                "evolstage_flame": 523.0,
                "phot_g_mean_mag": 11.0,
            }
        )
    return pd.DataFrame(rows)


def test_get_refinement_family_benchmark_task_returns_known_task() -> None:
    task = get_refinement_family_benchmark_task("gaia_mk_refinement_g_classification")

    assert task.target_column == "spectral_subclass"


def test_get_refinement_family_benchmark_task_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported refinement family benchmark task"):
        get_refinement_family_benchmark_task("gaia_mk_refinement_o_classification")


def test_run_refinement_family_benchmark_with_engine_returns_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "exohost.training.run_refinement_family_benchmark.load_refinement_family_prepared_training_frame",
        lambda engine, task_name, limit=None: build_family_frame(),
    )

    result = run_refinement_family_benchmark_with_engine(
        engine=Engine.__new__(Engine),
        task_name="gaia_mk_refinement_g_classification",
        limit=20,
        selected_model_names=("hist_gradient_boosting",),
    )

    assert result.task_name == "gaia_mk_refinement_g_classification"
    assert result.metrics_df["model_name"].tolist() == [
        "hist_gradient_boosting",
        "hist_gradient_boosting",
    ]
