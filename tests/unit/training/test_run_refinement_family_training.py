# Тестовый файл `test_run_refinement_family_training.py` домена `training`.
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

from exohost.training.run_refinement_family_training import (
    run_refinement_family_training_with_engine,
)


def build_family_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "spectral_subclass": "2",
                "teff_gspphot": 5800.0,
                "logg_gspphot": 4.4,
                "mh_gspphot": 0.1,
                "bp_rp": 0.75,
                "parallax": 15.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "radius_flame": 1.0,
                "lum_flame": 1.1,
                "evolstage_flame": 523.0,
                "phot_g_mean_mag": 10.1,
            },
            {
                "source_id": 2,
                "spectral_subclass": "4",
                "teff_gspphot": 5600.0,
                "logg_gspphot": 4.3,
                "mh_gspphot": 0.0,
                "bp_rp": 0.82,
                "parallax": 14.0,
                "parallax_over_error": 17.0,
                "ruwe": 1.02,
                "radius_flame": 0.9,
                "lum_flame": 1.0,
                "evolstage_flame": 524.0,
                "phot_g_mean_mag": 10.5,
            },
            {
                "source_id": 3,
                "spectral_subclass": "2",
                "teff_gspphot": 5790.0,
                "logg_gspphot": 4.4,
                "mh_gspphot": 0.1,
                "bp_rp": 0.76,
                "parallax": 16.0,
                "parallax_over_error": 19.0,
                "ruwe": 1.01,
                "radius_flame": 1.1,
                "lum_flame": 1.2,
                "evolstage_flame": 523.0,
                "phot_g_mean_mag": 10.0,
            },
            {
                "source_id": 4,
                "spectral_subclass": "4",
                "teff_gspphot": 5590.0,
                "logg_gspphot": 4.3,
                "mh_gspphot": -0.1,
                "bp_rp": 0.84,
                "parallax": 13.0,
                "parallax_over_error": 16.0,
                "ruwe": 1.03,
                "radius_flame": 0.8,
                "lum_flame": 0.9,
                "evolstage_flame": 524.0,
                "phot_g_mean_mag": 10.7,
            },
        ]
    )


def test_run_refinement_family_training_with_engine_returns_train_result(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "exohost.training.run_refinement_family_training.load_refinement_family_prepared_training_frame",
        lambda engine, task_name, limit=None: build_family_frame(),
    )

    result = run_refinement_family_training_with_engine(
        engine=cast(Engine, object()),
        task_name="gaia_mk_refinement_g_classification",
        model_name="hist_gradient_boosting",
        limit=100,
    )

    assert result.task_name == "gaia_mk_refinement_g_classification"
    assert result.model_name == "hist_gradient_boosting"
    assert result.n_rows == 4
