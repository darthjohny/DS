# Тестовый файл `test_run_hierarchical_training.py` домена `training`.
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

from exohost.training.run_hierarchical_training import (
    run_hierarchical_training_with_engine,
)


def build_coarse_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "teff_gspphot": 5800.0,
                "logg_gspphot": 4.4,
                "mh_gspphot": 0.1,
                "bp_rp": 0.75,
                "parallax": 15.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "radius_feature": 1.0,
                "spec_class": "G",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 2,
                "teff_gspphot": 4500.0,
                "logg_gspphot": 4.6,
                "mh_gspphot": -0.1,
                "bp_rp": 1.10,
                "parallax": 12.0,
                "parallax_over_error": 17.0,
                "ruwe": 1.02,
                "radius_feature": 0.8,
                "spec_class": "K",
                "evolution_stage": "evolved",
            },
            {
                "source_id": 3,
                "teff_gspphot": 5900.0,
                "logg_gspphot": 4.3,
                "mh_gspphot": 0.0,
                "bp_rp": 0.70,
                "parallax": 14.0,
                "parallax_over_error": 19.0,
                "ruwe": 1.03,
                "radius_feature": 1.1,
                "spec_class": "G",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 4,
                "teff_gspphot": 4400.0,
                "logg_gspphot": 4.7,
                "mh_gspphot": -0.2,
                "bp_rp": 1.15,
                "parallax": 11.0,
                "parallax_over_error": 16.0,
                "ruwe": 1.01,
                "radius_feature": 0.7,
                "spec_class": "K",
                "evolution_stage": "evolved",
            },
        ]
    )


def test_run_hierarchical_training_with_engine_returns_train_result(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "exohost.training.run_hierarchical_training.load_hierarchical_prepared_training_frame",
        lambda engine, task_name, limit=None: build_coarse_frame(),
    )

    result = run_hierarchical_training_with_engine(
        engine=cast(Engine, object()),
        task_name="gaia_id_coarse_classification",
        model_name="hist_gradient_boosting",
        limit=100,
    )

    assert result.task_name == "gaia_id_coarse_classification"
    assert result.model_name == "hist_gradient_boosting"
    assert result.n_rows == 4
