# Unit-тесты review-хелперов для feature separability `O vs B`.

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import pandas as pd

from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.protocol import ModelSpec
from exohost.reporting.archive_research.coarse_ob_feature_separability_review import (
    CoarseOBFeatureSeparabilityConfig,
    build_boundary_feature_physics_frame,
    build_boundary_membership_summary_frame,
    build_boundary_predicted_class_summary_frame,
    build_boundary_probability_summary_frame,
    build_boundary_true_class_summary_frame,
    build_coarse_ob_feature_separability_review_bundle,
    build_train_time_ob_boundary_frame,
    build_univariate_separability_auc_frame,
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


def _build_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4, 5],
            "spec_class": ["O", "O", "B", "B", "A"],
            "evolution_stage": ["evolved", "evolved", "dwarf", "dwarf", "dwarf"],
            "teff_gspphot": [32000.0, 30000.0, 18000.0, 16000.0, 9000.0],
            "logg_gspphot": [4.0, 4.1, 4.2, 4.0, 4.3],
            "mh_gspphot": [0.0, -0.1, 0.1, -0.1, 0.0],
            "bp_rp": [-0.25, -0.20, 0.0, 0.1, 0.2],
            "parallax": [1.9, 2.1, 2.2, 2.4, 2.6],
            "parallax_over_error": [16.0, 15.0, 18.0, 17.0, 19.0],
            "ruwe": [1.0, 1.0, 1.0, 1.0, 1.0],
            "radius_feature": [8.2, 7.4, 5.1, 4.7, 3.5],
        }
    )


def test_train_time_ob_boundary_frame_keeps_only_hot_o_and_b() -> None:
    boundary_df = build_train_time_ob_boundary_frame(
        _build_source_df(),
        config=CoarseOBFeatureSeparabilityConfig(hot_teff_min_k=10_000.0),
    )

    assert list(boundary_df["source_id"]) == [1, 2, 3, 4]
    assert set(boundary_df["spec_class"]) == {"O", "B"}


def test_feature_separability_bundle_builds_auc_probability_and_importance_frames(
    tmp_path: Path,
) -> None:
    source_df = _build_source_df()
    model_paths = save_model_artifacts(
        _build_train_result(_build_coarse_training_frame()),
        output_dir=tmp_path / "models",
    )

    bundle = build_coarse_ob_feature_separability_review_bundle(
        source_df,
        coarse_model_run_dir=model_paths.run_dir,
        config=CoarseOBFeatureSeparabilityConfig(
            hot_teff_min_k=10_000.0,
            permutation_n_repeats=5,
        ),
    )

    membership_df = build_boundary_membership_summary_frame(bundle)
    true_class_df = build_boundary_true_class_summary_frame(bundle.boundary_df)
    predicted_class_df = build_boundary_predicted_class_summary_frame(bundle.scored_boundary_df)
    probability_df = build_boundary_probability_summary_frame(bundle.scored_boundary_df)
    physics_df = build_boundary_feature_physics_frame(bundle.boundary_df)
    auc_df = build_univariate_separability_auc_frame(bundle.boundary_df)
    permutation_df = bundle.permutation_importance_df

    assert _require_int_scalar(membership_df.loc[0, "n_rows_boundary"]) == 4
    assert set(true_class_df["true_spectral_class"]) == {"O", "B"}
    assert _require_int_scalar(predicted_class_df["n_rows"].sum()) == 4
    assert set(probability_df["true_spectral_class"]) == {"O", "B"}
    assert set(physics_df["true_spectral_class"]) == {"O", "B"}
    assert "teff_gspphot" in auc_df["feature_name"].tolist()
    teff_auc = float(
        auc_df.loc[auc_df["feature_name"] == "teff_gspphot", "separability_auc"].iloc[0]
    )
    assert teff_auc > 0.9
    assert set(permutation_df["metric_name"]) == {"accuracy", "balanced_accuracy"}
    assert "teff_gspphot" in permutation_df["feature_name"].tolist()
