# Тестовый файл `test_cli_decide.py` домена `cli`.
#
# Этот файл проверяет только:
# - проверку логики домена: CLI-команды и их orchestration-сценарии;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `cli` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from exohost.cli.main import main
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
from exohost.training.train_runner import run_training


def build_ood_training_frame() -> pd.DataFrame:
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


def build_train_result(frame: pd.DataFrame, task, *, model_name: str = "hist_gradient_boosting"):
    return run_training(
        frame,
        task=task,
        model_spec=ModelSpec(
            model_name=model_name,
            estimator=HGBClassifier(
                feature_columns=task.feature_columns,
                max_iter=20,
                min_samples_leaf=1,
                model_name=model_name,
            ),
        ),
    )


def test_cli_decide_builds_final_decision_artifacts(tmp_path: Path) -> None:
    ood_paths = save_model_artifacts(
        build_train_result(build_ood_training_frame(), GAIA_ID_OOD_CLASSIFICATION_TASK),
        output_dir=tmp_path / "models",
    )
    coarse_paths = save_model_artifacts(
        build_train_result(build_coarse_training_frame(), GAIA_ID_COARSE_CLASSIFICATION_TASK),
        output_dir=tmp_path / "models",
    )
    host_paths = save_model_artifacts(
        build_train_result(build_host_training_frame(), HOST_FIELD_CLASSIFICATION_TASK),
        output_dir=tmp_path / "models",
    )
    threshold_paths = save_id_ood_threshold_artifact(
        build_id_ood_threshold_policy(
            tuned_threshold=0.65,
            threshold_policy_version="id_ood_threshold_v1",
            candidate_ood_threshold=0.45,
        ),
        task_name=GAIA_ID_OOD_CLASSIFICATION_TASK.name,
        model_name="hist_gradient_boosting",
        output_dir=tmp_path / "thresholds",
    )

    input_csv_path = tmp_path / "decision_input.csv"
    output_dir = tmp_path / "decision_artifacts"
    pd.DataFrame(
        [
            {
                "source_id": "501",
                "quality_state": "pass",
                "teff_gspphot": 5810.0,
                "logg_gspphot": 4.4,
                "mh_gspphot": 0.08,
                "bp_rp": 0.74,
                "parallax": 15.1,
                "parallax_over_error": 18.2,
                "ruwe": 1.01,
                "phot_g_mean_mag": 10.9,
                "radius_feature": 1.0,
                "radius_gspphot": 1.0,
                "validation_factor": 0.92,
            },
            {
                "source_id": "502",
                "quality_state": "pass",
                "teff_gspphot": 4460.0,
                "logg_gspphot": 4.7,
                "mh_gspphot": -0.18,
                "bp_rp": 1.13,
                "parallax": 10.0,
                "parallax_over_error": 15.0,
                "ruwe": 1.04,
                "phot_g_mean_mag": 13.1,
                "radius_feature": 0.8,
                "radius_gspphot": 0.8,
                "validation_factor": 0.61,
            },
        ]
    ).to_csv(input_csv_path, index=False)

    exit_code = main(
        [
            "decide",
            "--input-csv",
            str(input_csv_path),
            "--ood-model-run-dir",
            str(ood_paths.run_dir),
            "--ood-threshold-run-dir",
            str(threshold_paths.run_dir),
            "--coarse-model-run-dir",
            str(coarse_paths.run_dir),
            "--host-model-run-dir",
            str(host_paths.run_dir),
            "--priority-high-min",
            "0.85",
            "--priority-medium-min",
            "0.55",
            "--output-dir",
            str(output_dir),
            "--preview-rows",
            "2",
        ]
    )

    assert exit_code == 0

    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "decision_input.csv").exists()
    assert (run_dir / "final_decision.csv").exists()
    assert (run_dir / "priority_input.csv").exists()
    assert (run_dir / "priority_ranking.csv").exists()
    assert (run_dir / "metadata.json").exists()

    final_decision_df = pd.read_csv(run_dir / "final_decision.csv")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "final_domain_state" in final_decision_df.columns
    assert "final_decision_reason" in final_decision_df.columns
    assert metadata["context"]["priority_high_min"] == 0.85
    assert metadata["context"]["priority_medium_min"] == 0.55
