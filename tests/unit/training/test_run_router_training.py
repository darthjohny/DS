# Тестовый файл `test_run_router_training.py` домена `training`.
#
# Этот файл проверяет только:
# - проверку логики домена: обучающие orchestration-сценарии и benchmark-runner;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `training` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from typing import cast

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.training.run_router_training import run_router_training_with_engine


def build_router_frame() -> pd.DataFrame:
    # Synthetic router dataset для training-оркестрации.
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "ra": 10.0,
                "dec": 20.0,
                "teff_gspphot": 5800.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 1.0,
                "parallax": 15.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "bp_rp": 0.75,
                "mh_gspphot": 0.1,
                "spec_class": "G",
                "spec_subclass": "G2",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 2,
                "ra": 11.0,
                "dec": 21.0,
                "teff_gspphot": 4500.0,
                "logg_gspphot": 4.6,
                "radius_gspphot": 0.8,
                "parallax": 12.0,
                "parallax_over_error": 17.0,
                "ruwe": 1.02,
                "bp_rp": 1.10,
                "mh_gspphot": -0.1,
                "spec_class": "K",
                "spec_subclass": "K4",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 3,
                "ra": 12.0,
                "dec": 22.0,
                "teff_gspphot": 5900.0,
                "logg_gspphot": 4.3,
                "radius_gspphot": 1.1,
                "parallax": 14.0,
                "parallax_over_error": 19.0,
                "ruwe": 1.03,
                "bp_rp": 0.70,
                "mh_gspphot": 0.0,
                "spec_class": "G",
                "spec_subclass": "G1",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 4,
                "ra": 13.0,
                "dec": 23.0,
                "teff_gspphot": 4400.0,
                "logg_gspphot": 4.7,
                "radius_gspphot": 0.7,
                "parallax": 11.0,
                "parallax_over_error": 16.0,
                "ruwe": 1.01,
                "bp_rp": 1.15,
                "mh_gspphot": -0.2,
                "spec_class": "K",
                "spec_subclass": "K5",
                "evolution_stage": "dwarf",
            },
        ]
    )


def test_run_router_training_with_engine_returns_train_result(monkeypatch) -> None:
    # Проверяем router training-поток без реального подключения к БД.
    monkeypatch.setattr(
        "exohost.training.run_router_training.load_router_training_dataset",
        lambda engine, limit=None: build_router_frame(),
    )

    result = run_router_training_with_engine(
        engine=cast(Engine, object()),
        task_name="spectral_class_classification",
        model_name="hist_gradient_boosting",
        limit=100,
    )

    assert result.task_name == "spectral_class_classification"
    assert result.model_name == "hist_gradient_boosting"
    assert result.n_rows == 4
