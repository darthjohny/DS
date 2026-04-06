# Testkit для малого decision-regression контура.
#
# Этот файл отвечает только за:
# - подготовку маленьких synthetic artifacts для regression-тестов `decide`;
# - сборку путей и входов для сквозного decision roundtrip.
#
# Следующий слой:
# - regression-тесты `decision`-подпакета;
# - CLI `decide` и artifact bundle, которые они страхуют.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfile

import pandas as pd

from exohost.evaluation.hierarchical_tasks import (
    GAIA_ID_COARSE_CLASSIFICATION_TASK,
    GAIA_ID_OOD_CLASSIFICATION_TASK,
)
from exohost.evaluation.protocol import HOST_FIELD_CLASSIFICATION_TASK
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.protocol import ModelSpec
from exohost.posthoc.id_ood_gate import build_id_ood_threshold_policy
from exohost.reporting.id_ood_threshold_artifacts import save_id_ood_threshold_artifact
from exohost.reporting.model_artifacts import save_model_artifacts
from exohost.training.train_runner import TrainRunResult, run_training
from tests.regression.conftest import DECIDE_INPUT_SMALL_FIXTURE_PATH

_FIXED_ARTIFACT_TIME = datetime(2026, 4, 6, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class DecisionRegressionPaths:
    # Пути к подготовленным artifacts и frozen input для малого decide-regression.
    input_csv_path: Path
    output_dir: Path
    ood_model_run_dir: Path
    ood_threshold_run_dir: Path
    coarse_model_run_dir: Path
    host_model_run_dir: Path


def prepare_decision_regression_paths(tmp_path: Path) -> DecisionRegressionPaths:
    # Поднимаем минимальный набор trained artifacts и frozen decision input.
    models_dir = tmp_path / "models"
    thresholds_dir = tmp_path / "thresholds"
    output_dir = tmp_path / "decision_artifacts"
    input_csv_path = tmp_path / "decision_input.csv"

    copyfile(DECIDE_INPUT_SMALL_FIXTURE_PATH, input_csv_path)

    ood_paths = save_model_artifacts(
        _build_train_result(build_ood_training_frame(), GAIA_ID_OOD_CLASSIFICATION_TASK),
        output_dir=models_dir,
        now=_FIXED_ARTIFACT_TIME,
    )
    coarse_paths = save_model_artifacts(
        _build_train_result(build_coarse_training_frame(), GAIA_ID_COARSE_CLASSIFICATION_TASK),
        output_dir=models_dir,
        now=_FIXED_ARTIFACT_TIME.replace(minute=1),
    )
    host_paths = save_model_artifacts(
        _build_train_result(build_host_training_frame(), HOST_FIELD_CLASSIFICATION_TASK),
        output_dir=models_dir,
        now=_FIXED_ARTIFACT_TIME.replace(minute=2),
    )
    threshold_paths = save_id_ood_threshold_artifact(
        build_id_ood_threshold_policy(
            tuned_threshold=0.65,
            threshold_policy_version="id_ood_threshold_v1",
            candidate_ood_threshold=0.45,
        ),
        task_name=GAIA_ID_OOD_CLASSIFICATION_TASK.name,
        model_name="hist_gradient_boosting",
        output_dir=thresholds_dir,
        now=_FIXED_ARTIFACT_TIME.replace(minute=3),
    )

    return DecisionRegressionPaths(
        input_csv_path=input_csv_path,
        output_dir=output_dir,
        ood_model_run_dir=ood_paths.run_dir,
        ood_threshold_run_dir=threshold_paths.run_dir,
        coarse_model_run_dir=coarse_paths.run_dir,
        host_model_run_dir=host_paths.run_dir,
    )


def build_ood_training_frame() -> pd.DataFrame:
    # Малый synthetic train frame для ID/OOD слоя.
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
                "phot_g_mean_mag": 10.8,
                "domain_target": "id",
            },
            {
                "source_id": 2,
                "teff_gspphot": 4700.0,
                "logg_gspphot": 4.7,
                "mh_gspphot": -0.2,
                "bp_rp": 1.20,
                "parallax": 8.0,
                "parallax_over_error": 6.0,
                "ruwe": 1.40,
                "phot_g_mean_mag": 14.0,
                "domain_target": "ood",
            },
            {
                "source_id": 3,
                "teff_gspphot": 5900.0,
                "logg_gspphot": 4.3,
                "mh_gspphot": 0.0,
                "bp_rp": 0.70,
                "parallax": 14.0,
                "parallax_over_error": 17.0,
                "ruwe": 1.02,
                "phot_g_mean_mag": 11.0,
                "domain_target": "id",
            },
            {
                "source_id": 4,
                "teff_gspphot": 4300.0,
                "logg_gspphot": 4.8,
                "mh_gspphot": -0.3,
                "bp_rp": 1.35,
                "parallax": 7.5,
                "parallax_over_error": 5.5,
                "ruwe": 1.50,
                "phot_g_mean_mag": 14.2,
                "domain_target": "ood",
            },
        ]
    )


def build_coarse_training_frame() -> pd.DataFrame:
    # Малый synthetic train frame для coarse spectral class.
    return pd.DataFrame(
        [
            {
                "source_id": 10,
                "teff_gspphot": 5800.0,
                "logg_gspphot": 4.4,
                "mh_gspphot": 0.1,
                "bp_rp": 0.75,
                "parallax": 15.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "radius_feature": 1.0,
                "spec_class": "G",
            },
            {
                "source_id": 11,
                "teff_gspphot": 4500.0,
                "logg_gspphot": 4.7,
                "mh_gspphot": -0.2,
                "bp_rp": 1.10,
                "parallax": 11.0,
                "parallax_over_error": 16.0,
                "ruwe": 1.02,
                "radius_feature": 0.8,
                "spec_class": "K",
            },
            {
                "source_id": 12,
                "teff_gspphot": 5900.0,
                "logg_gspphot": 4.3,
                "mh_gspphot": 0.0,
                "bp_rp": 0.70,
                "parallax": 14.0,
                "parallax_over_error": 19.0,
                "ruwe": 1.03,
                "radius_feature": 1.1,
                "spec_class": "G",
            },
            {
                "source_id": 13,
                "teff_gspphot": 4400.0,
                "logg_gspphot": 4.8,
                "mh_gspphot": -0.1,
                "bp_rp": 1.15,
                "parallax": 10.0,
                "parallax_over_error": 15.0,
                "ruwe": 1.04,
                "radius_feature": 0.7,
                "spec_class": "K",
            },
        ]
    )


def build_host_training_frame() -> pd.DataFrame:
    # Малый synthetic train frame для host-vs-field модели.
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
                "host_label": "field",
            },
        ]
    )


def _build_train_result(frame: pd.DataFrame, task) -> TrainRunResult:
    return run_training(
        frame,
        task=task,
        model_spec=ModelSpec(
            model_name="hist_gradient_boosting",
            estimator=HGBClassifier(
                feature_columns=task.feature_columns,
                max_iter=20,
                min_samples_leaf=1,
                model_name="hist_gradient_boosting",
            ),
        ),
    )

