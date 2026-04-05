# Тестовый файл `test_coarse_ob_domain_shift_review.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import pandas as pd

from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.protocol import ModelSpec
from exohost.reporting.coarse_ob_domain_shift_review import (
    CoarseOBDomainShiftConfig,
    build_coarse_ob_domain_shift_review_bundle,
    build_domain_class_balance_frame,
    build_domain_confusion_frame,
    build_domain_membership_summary_frame,
    build_domain_missingness_summary_frame,
    build_domain_physics_summary_frame,
    build_domain_probability_summary_frame,
    build_domain_shift_auc_frame,
)
from exohost.reporting.model_artifacts import save_model_artifacts
from exohost.training.train_runner import TrainRunResult, run_training


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _build_coarse_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 10,
                "teff_gspphot": 34000.0,
                "logg_gspphot": 4.0,
                "mh_gspphot": 0.0,
                "bp_rp": -0.2,
                "parallax": 1.8,
                "parallax_over_error": 15.0,
                "ruwe": 1.0,
                "radius_feature": 8.0,
                "spec_class": "O",
            },
            {
                "source_id": 11,
                "teff_gspphot": 32000.0,
                "logg_gspphot": 4.1,
                "mh_gspphot": -0.1,
                "bp_rp": -0.1,
                "parallax": 1.9,
                "parallax_over_error": 14.0,
                "ruwe": 1.0,
                "radius_feature": 7.2,
                "spec_class": "O",
            },
            {
                "source_id": 12,
                "teff_gspphot": 18000.0,
                "logg_gspphot": 4.2,
                "mh_gspphot": 0.1,
                "bp_rp": 0.0,
                "parallax": 2.3,
                "parallax_over_error": 18.0,
                "ruwe": 1.0,
                "radius_feature": 5.1,
                "spec_class": "B",
            },
            {
                "source_id": 13,
                "teff_gspphot": 16000.0,
                "logg_gspphot": 4.1,
                "mh_gspphot": -0.1,
                "bp_rp": 0.1,
                "parallax": 2.5,
                "parallax_over_error": 17.0,
                "ruwe": 1.0,
                "radius_feature": 4.9,
                "spec_class": "B",
            },
        ]
    )


def _build_train_result(frame: pd.DataFrame) -> TrainRunResult:
    return run_training(
        frame,
        task=GAIA_ID_COARSE_CLASSIFICATION_TASK,
        model_spec=ModelSpec(
            model_name="hist_gradient_boosting",
            estimator=HGBClassifier(
                feature_columns=GAIA_ID_COARSE_CLASSIFICATION_TASK.feature_columns,
                max_iter=20,
                min_samples_leaf=1,
                model_name="hist_gradient_boosting",
            ),
        ),
    )


def _build_train_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4, 5],
            "spec_class": ["O", "O", "B", "B", "A"],
            "evolution_stage": ["evolved", "evolved", "dwarf", "dwarf", "dwarf"],
            "teff_gspphot": [34000.0, 32000.0, 18000.0, 16000.0, 9000.0],
            "logg_gspphot": [4.0, 4.1, 4.2, 4.0, 4.3],
            "mh_gspphot": [0.0, -0.1, 0.1, -0.1, 0.0],
            "bp_rp": [-0.25, -0.20, 0.0, 0.1, 0.2],
            "parallax": [1.8, 1.9, 2.3, 2.5, 2.6],
            "parallax_over_error": [16.0, 15.0, 18.0, 17.0, 19.0],
            "ruwe": [1.0, 1.0, 1.0, 1.0, 1.0],
            "radius_feature": [8.2, 7.4, 5.1, 4.7, 3.5],
        }
    )


def _build_downstream_boundary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [101, 102, 103, 104],
            "spectral_class": ["O", "O", "B", "B"],
            "quality_state": ["pass", "pass", "pass", "pass"],
            "quality_reason": ["pass", "pass", "pass", "pass"],
            "review_bucket": ["pass", "pass", "pass", "pass"],
            "teff_gspphot": [29000.0, 28000.0, 17500.0, 16500.0],
            "logg_gspphot": [3.8, 3.9, 4.0, 4.1],
            "mh_gspphot": [0.2, 0.1, -0.2, -0.1],
            "bp_rp": [0.2, 0.25, 0.05, 0.1],
            "parallax": [1.2, 1.4, 2.0, 2.1],
            "parallax_over_error": [8.0, 9.0, 13.0, 12.0],
            "ruwe": [1.1, 1.1, 1.0, 1.0],
            "radius_flame": [7.0, 6.8, 5.0, 4.8],
        }
    )


def test_coarse_ob_domain_shift_bundle_builds_compare_frames(tmp_path: Path) -> None:
    model_paths = save_model_artifacts(
        _build_train_result(_build_coarse_training_frame()),
        output_dir=tmp_path / "models",
    )

    bundle = build_coarse_ob_domain_shift_review_bundle(
        _build_train_source_df(),
        downstream_boundary_df=_build_downstream_boundary_df(),
        coarse_model_run_dir=model_paths.run_dir,
        config=CoarseOBDomainShiftConfig(quality_state="pass", hot_teff_min_k=10_000.0),
    )

    membership_df = build_domain_membership_summary_frame(bundle)
    class_balance_df = build_domain_class_balance_frame(bundle)
    confusion_df = build_domain_confusion_frame(bundle)
    probability_df = build_domain_probability_summary_frame(bundle)
    physics_df = build_domain_physics_summary_frame(bundle)
    missingness_df = build_domain_missingness_summary_frame(bundle)
    auc_df = build_domain_shift_auc_frame(bundle)

    assert _require_int_scalar(membership_df.loc[0, "n_rows_train_boundary"]) == 4
    assert _require_int_scalar(membership_df.loc[0, "n_rows_downstream_boundary"]) == 4
    assert set(class_balance_df["domain_name"]) == {"train_time", "downstream_pass"}
    assert set(confusion_df["domain_name"]) == {"train_time", "downstream_pass"}
    assert set(probability_df["domain_name"]) == {"train_time", "downstream_pass"}
    assert set(physics_df["domain_name"]) == {"train_time", "downstream_pass"}
    assert "radius_feature" in missingness_df["feature_name"].tolist()
    assert set(auc_df["true_spectral_class"]) == {"O", "B"}
