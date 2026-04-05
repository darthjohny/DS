# Тестовый файл `test_train_runner.py` домена `training`.
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

from exohost.evaluation.protocol import SPECTRAL_CLASS_CLASSIFICATION_TASK
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.protocol import ModelSpec
from exohost.training.train_runner import run_training


def build_training_frame() -> pd.DataFrame:
    # Небольшой synthetic dataset для train runner.
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "teff_gspphot": 5800.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 1.0,
                "parallax": 15.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "bp_rp": 0.75,
                "mh_gspphot": 0.1,
                "spec_class": "G",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 2,
                "teff_gspphot": 4500.0,
                "logg_gspphot": 4.6,
                "radius_gspphot": 0.8,
                "parallax": 12.0,
                "parallax_over_error": 17.0,
                "ruwe": 1.02,
                "bp_rp": 1.10,
                "mh_gspphot": -0.1,
                "spec_class": "K",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 3,
                "teff_gspphot": 5900.0,
                "logg_gspphot": 4.3,
                "radius_gspphot": 1.1,
                "parallax": 14.0,
                "parallax_over_error": 19.0,
                "ruwe": 1.03,
                "bp_rp": 0.70,
                "mh_gspphot": 0.0,
                "spec_class": "G",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 4,
                "teff_gspphot": 4400.0,
                "logg_gspphot": 4.7,
                "radius_gspphot": 0.7,
                "parallax": 11.0,
                "parallax_over_error": 16.0,
                "ruwe": 1.01,
                "bp_rp": 1.15,
                "mh_gspphot": -0.2,
                "spec_class": "K",
                "evolution_stage": "dwarf",
            },
        ]
    )


def test_run_training_returns_fitted_estimator_and_distribution() -> None:
    # Проверяем, что train runner обучает модель и собирает metadata-контур.
    task = SPECTRAL_CLASS_CLASSIFICATION_TASK
    model_spec = ModelSpec(
        model_name="hist_gradient_boosting",
        estimator=HGBClassifier(
            feature_columns=task.feature_columns,
            max_iter=20,
            min_samples_leaf=1,
            model_name="hist_gradient_boosting",
        ),
    )

    result = run_training(
        build_training_frame(),
        task=task,
        model_spec=model_spec,
    )

    assert result.task_name == task.name
    assert result.model_name == "hist_gradient_boosting"
    assert result.n_rows == 4
    assert result.class_labels == ("G", "K")
    assert set(result.label_distribution_df["target_label"]) == {"G", "K"}
