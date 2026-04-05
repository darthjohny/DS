# Общие pytest-fixtures для повторно используемых synthetic наборов.

from __future__ import annotations

import pandas as pd
import pytest

from exohost.evaluation.protocol import (
    HOST_FIELD_CLASSIFICATION_TASK,
    SPECTRAL_CLASS_CLASSIFICATION_TASK,
)
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.protocol import ModelSpec
from exohost.training.train_runner import TrainRunResult, run_training


@pytest.fixture
def small_spectral_class_training_frame() -> pd.DataFrame:
    # Базовый synthetic train frame для задач по coarse spectral class.
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


@pytest.fixture
def small_model_scoring_frame() -> pd.DataFrame:
    # Базовый synthetic scoring frame для применения сохраненной модели.
    return pd.DataFrame(
        [
            {
                "source_id": "10",
                "teff_gspphot": 5820.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 1.0,
                "parallax": 14.5,
                "parallax_over_error": 19.0,
                "ruwe": 1.01,
                "bp_rp": 0.76,
                "mh_gspphot": 0.05,
            },
            {
                "source_id": "11",
                "teff_gspphot": 4450.0,
                "logg_gspphot": 4.7,
                "radius_gspphot": 0.8,
                "parallax": 11.5,
                "parallax_over_error": 16.5,
                "ruwe": 1.02,
                "bp_rp": 1.12,
                "mh_gspphot": -0.15,
            },
        ]
    )


@pytest.fixture
def small_spectral_class_train_result(
    small_spectral_class_training_frame: pd.DataFrame,
) -> TrainRunResult:
    # Базовый fitted train-result для CLI и artifact-тестов.
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
    return run_training(
        small_spectral_class_training_frame,
        task=task,
        model_spec=model_spec,
    )


@pytest.fixture
def small_host_field_training_frame() -> pd.DataFrame:
    # Базовый synthetic train frame для host-vs-field модели.
    return pd.DataFrame(
        [
            {
                "source_id": 101,
                "teff_gspphot": 5850.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 1.0,
                "parallax": 15.0,
                "parallax_over_error": 20.0,
                "ruwe": 1.01,
                "bp_rp": 0.74,
                "mh_gspphot": 0.08,
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "host_label": "host",
            },
            {
                "source_id": 102,
                "teff_gspphot": 5750.0,
                "logg_gspphot": 4.5,
                "radius_gspphot": 0.98,
                "parallax": 13.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.03,
                "bp_rp": 0.77,
                "mh_gspphot": 0.02,
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "host_label": "field",
            },
            {
                "source_id": 103,
                "teff_gspphot": 4520.0,
                "logg_gspphot": 4.6,
                "radius_gspphot": 0.82,
                "parallax": 11.0,
                "parallax_over_error": 16.0,
                "ruwe": 1.02,
                "bp_rp": 1.09,
                "mh_gspphot": -0.12,
                "spec_class": "K",
                "evolution_stage": "dwarf",
                "host_label": "host",
            },
            {
                "source_id": 104,
                "teff_gspphot": 4460.0,
                "logg_gspphot": 4.7,
                "radius_gspphot": 0.78,
                "parallax": 10.0,
                "parallax_over_error": 15.0,
                "ruwe": 1.04,
                "bp_rp": 1.13,
                "mh_gspphot": -0.18,
                "spec_class": "K",
                "evolution_stage": "dwarf",
                "host_label": "field",
            },
        ]
    )


@pytest.fixture
def small_host_field_train_result(
    small_host_field_training_frame: pd.DataFrame,
) -> TrainRunResult:
    # Базовый fitted train-result для host scoring и ranking-контура.
    task = HOST_FIELD_CLASSIFICATION_TASK
    model_spec = ModelSpec(
        model_name="hist_gradient_boosting",
        estimator=HGBClassifier(
            feature_columns=task.feature_columns,
            max_iter=20,
            min_samples_leaf=1,
            model_name="hist_gradient_boosting",
        ),
    )
    return run_training(
        small_host_field_training_frame,
        task=task,
        model_spec=model_spec,
    )
