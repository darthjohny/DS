# Unit-тесты narrow review-хелперов для границы `O/B`.

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import pandas as pd

from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.protocol import ModelSpec
from exohost.reporting.archive_research.coarse_ob_boundary_review import (
    CoarseOBBoundaryReviewConfig,
    build_boundary_confusion_frame,
    build_boundary_membership_summary_frame,
    build_boundary_physics_summary_frame,
    build_boundary_predicted_class_summary_frame,
    build_boundary_probability_summary_frame,
    build_boundary_true_class_summary_frame,
    build_coarse_ob_boundary_review_bundle,
    build_high_confidence_b_to_o_preview_frame,
    build_high_confidence_o_to_b_preview_frame,
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
                "teff_gspphot": 33000.0,
                "logg_gspphot": 4.1,
                "mh_gspphot": 0.0,
                "bp_rp": -0.2,
                "parallax": 2.0,
                "parallax_over_error": 15.0,
                "ruwe": 1.0,
                "radius_feature": 8.0,
                "spec_class": "O",
            },
            {
                "source_id": 11,
                "teff_gspphot": 30000.0,
                "logg_gspphot": 4.0,
                "mh_gspphot": -0.1,
                "bp_rp": -0.1,
                "parallax": 1.8,
                "parallax_over_error": 14.0,
                "ruwe": 1.0,
                "radius_feature": 7.0,
                "spec_class": "O",
            },
            {
                "source_id": 12,
                "teff_gspphot": 18000.0,
                "logg_gspphot": 4.2,
                "mh_gspphot": 0.1,
                "bp_rp": 0.0,
                "parallax": 2.2,
                "parallax_over_error": 18.0,
                "ruwe": 1.0,
                "radius_feature": 5.0,
                "spec_class": "B",
            },
            {
                "source_id": 13,
                "teff_gspphot": 15000.0,
                "logg_gspphot": 4.1,
                "mh_gspphot": -0.1,
                "bp_rp": 0.1,
                "parallax": 2.4,
                "parallax_over_error": 17.0,
                "ruwe": 1.0,
                "radius_feature": 4.8,
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


def _build_boundary_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "spectral_class": ["O", "O", "B", "B"],
            "quality_state": ["pass", "pass", "pass", "pass"],
            "quality_reason": ["pass", "pass", "pass", "pass"],
            "review_bucket": ["pass", "pass", "pass", "pass"],
            "ood_state": ["in_domain", "in_domain", "in_domain", "in_domain"],
            "ood_reason": ["in_domain", "in_domain", "in_domain", "in_domain"],
            "teff_gspphot": [32000.0, 30000.0, 18000.0, 16000.0],
            "logg_gspphot": [4.0, 4.1, 4.2, 4.0],
            "mh_gspphot": [0.0, -0.1, 0.1, -0.1],
            "bp_rp": [-0.25, -0.20, 0.0, 0.1],
            "parallax": [1.9, 2.1, 2.2, 2.4],
            "parallax_over_error": [16.0, 15.0, 18.0, 17.0],
            "ruwe": [1.0, 1.0, 1.0, 1.0],
            "radius_flame": [8.2, 7.4, 5.1, 4.7],
            "random_index": [101, 102, 103, 104],
        }
    )


def test_boundary_review_bundle_builds_probability_summary_and_confusion(
    tmp_path: Path,
) -> None:
    source_df = _build_boundary_source_df()
    model_paths = save_model_artifacts(
        _build_train_result(_build_coarse_training_frame()),
        output_dir=tmp_path / "models",
    )

    bundle = build_coarse_ob_boundary_review_bundle(
        source_df,
        coarse_model_run_dir=model_paths.run_dir,
        config=CoarseOBBoundaryReviewConfig(),
    )

    membership_df = build_boundary_membership_summary_frame(bundle)
    true_class_df = build_boundary_true_class_summary_frame(bundle.source_df)
    predicted_class_df = build_boundary_predicted_class_summary_frame(bundle.scored_df)
    confusion_df = build_boundary_confusion_frame(bundle.scored_df)
    probability_df = build_boundary_probability_summary_frame(bundle.scored_df)
    physics_df = build_boundary_physics_summary_frame(bundle.source_df)
    o_to_b_preview_df = build_high_confidence_o_to_b_preview_frame(bundle.scored_df)
    b_to_o_preview_df = build_high_confidence_b_to_o_preview_frame(bundle.scored_df)

    assert _require_int_scalar(membership_df.loc[0, "n_rows_boundary_source"]) == 4
    assert _require_int_scalar(membership_df.loc[0, "n_rows_scored"]) == 4
    assert set(true_class_df["true_spectral_class"]) == {"O", "B"}
    assert "coarse_predicted_label" in bundle.scored_df.columns
    assert "coarse_probability__O" in bundle.scored_df.columns
    assert "coarse_probability__B" in bundle.scored_df.columns
    assert _require_int_scalar(predicted_class_df["n_rows"].sum()) == 4
    assert _require_int_scalar(confusion_df["n_rows"].sum()) == 4
    assert set(probability_df["true_spectral_class"]) == {"O", "B"}
    assert "median_teff_gspphot" in physics_df.columns
    assert int(o_to_b_preview_df.shape[0]) <= 2
    assert int(b_to_o_preview_df.shape[0]) <= 2
