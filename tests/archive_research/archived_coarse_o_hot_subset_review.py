# Unit-тесты review-хелперов для physically hot `O/B-like` subset.

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import pandas as pd

from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.protocol import ModelSpec
from exohost.reporting.archive_research.coarse_o_hot_subset_review import (
    HotOBLikeSubsetConfig,
    build_hot_ob_like_subset_review_bundle,
    build_hot_ob_like_subset_source_frame,
    build_hot_subset_final_coarse_distribution_frame,
    build_hot_subset_final_outcome_distribution_frame,
    build_hot_subset_final_reason_frame,
    build_hot_subset_high_confidence_non_ob_preview_frame,
    build_hot_subset_membership_summary_frame,
    build_hot_subset_predicted_physics_summary_frame,
    build_hot_subset_quality_reason_frame,
    build_hot_subset_quality_summary_frame,
    build_hot_subset_scored_prediction_frame,
)
from exohost.reporting.final_decision_artifacts import save_final_decision_artifacts
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
                "teff_gspphot": 18000.0,
                "logg_gspphot": 4.0,
                "mh_gspphot": -0.1,
                "bp_rp": 0.0,
                "parallax": 1.8,
                "parallax_over_error": 14.0,
                "ruwe": 1.0,
                "radius_feature": 6.0,
                "spec_class": "B",
            },
            {
                "source_id": 12,
                "teff_gspphot": 5600.0,
                "logg_gspphot": 4.5,
                "mh_gspphot": 0.1,
                "bp_rp": 0.8,
                "parallax": 12.0,
                "parallax_over_error": 20.0,
                "ruwe": 1.0,
                "radius_feature": 1.0,
                "spec_class": "G",
            },
            {
                "source_id": 13,
                "teff_gspphot": 4700.0,
                "logg_gspphot": 4.7,
                "mh_gspphot": -0.1,
                "bp_rp": 1.2,
                "parallax": 10.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.0,
                "radius_feature": 0.8,
                "spec_class": "K",
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


def _build_o_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4, 5],
            "spectral_class": ["O", "O", "O", "O", "O"],
            "quality_state": ["pass", "pass", "pass", "unknown", "reject"],
            "quality_reason": [
                "pass",
                "pass",
                "pass",
                "high_ruwe",
                "missing_core_features",
            ],
            "review_bucket": [
                "pass",
                "pass",
                "pass",
                "review_high_ruwe",
                "reject_missing_core_features",
            ],
            "ood_state": ["in_domain", "in_domain", "in_domain", "in_domain", "in_domain"],
            "ood_reason": ["in_domain", "in_domain", "in_domain", "in_domain", "in_domain"],
            "teff_gspphot": [32000.0, 15000.0, 8200.0, 29500.0, pd.NA],
            "logg_gspphot": [4.0, 4.1, 4.0, 3.9, 3.8],
            "mh_gspphot": [0.0, -0.1, -0.2, -0.2, -0.1],
            "bp_rp": [-0.25, -0.10, 0.40, -0.18, 0.20],
            "parallax": [1.9, 2.1, 1.6, 1.5, 1.3],
            "parallax_over_error": [16.0, 15.0, 12.0, 4.0, 2.0],
            "ruwe": [1.0, 1.1, 1.0, 1.8, 1.2],
            "radius_flame": [8.2, 5.7, 3.0, 7.1, pd.NA],
            "random_index": [101, 102, 103, 104, 105],
        }
    )


def _build_final_decision_artifact(tmp_path: Path) -> Path:
    paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=pd.DataFrame(
            {
                "source_id": [1, 2, 3, 4, 5],
                "quality_reason": [
                    "pass",
                    "pass",
                    "pass",
                    "high_ruwe",
                    "missing_core_features",
                ],
                "review_bucket": [
                    "pass",
                    "pass",
                    "pass",
                    "review_high_ruwe",
                    "reject_missing_core_features",
                ],
                "ood_state": ["in_domain", "in_domain", "in_domain", "in_domain", "in_domain"],
                "ood_reason": ["in_domain", "in_domain", "in_domain", "in_domain", "in_domain"],
            }
        ),
        final_decision_df=pd.DataFrame(
            {
                "source_id": [1, 2, 3, 4, 5],
                "final_domain_state": ["id", "id", "id", "unknown", "unknown"],
                "final_quality_state": ["pass", "pass", "pass", "unknown", "reject"],
                "final_coarse_class": ["O", "B", "A", pd.NA, pd.NA],
                "final_coarse_confidence": [0.91, 0.66, 0.55, pd.NA, pd.NA],
                "final_refinement_label": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                "final_refinement_state": [
                    "not_attempted",
                    "not_attempted",
                    "not_attempted",
                    "not_attempted",
                    "not_attempted",
                ],
                "final_refinement_confidence": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                "final_decision_reason": [
                    "coarse_only",
                    "coarse_only",
                    "coarse_only",
                    "quality_unknown",
                    "quality_reject",
                ],
            }
        ),
        priority_input_df=pd.DataFrame(columns=["source_id"]),
        priority_ranking_df=pd.DataFrame(columns=["source_id"]),
        output_dir=tmp_path,
    )
    return paths.run_dir


def test_build_hot_ob_like_subset_source_frame_applies_teff_cut() -> None:
    source_df = _build_o_source_df()

    result = build_hot_ob_like_subset_source_frame(
        source_df,
        config=HotOBLikeSubsetConfig(teff_min_k=10_000.0),
    )

    assert list(result["source_id"]) == [1, 2, 4]


def test_hot_ob_like_review_bundle_scopes_to_hot_subset_and_final_outputs(
    tmp_path: Path,
) -> None:
    source_df = _build_o_source_df()
    model_paths = save_model_artifacts(
        _build_train_result(_build_coarse_training_frame()),
        output_dir=tmp_path / "models",
    )
    final_run_dir = _build_final_decision_artifact(tmp_path / "decisions")

    bundle = build_hot_ob_like_subset_review_bundle(
        source_df,
        coarse_model_run_dir=model_paths.run_dir,
        final_decision_run_dir=final_run_dir,
        config=HotOBLikeSubsetConfig(teff_min_k=10_000.0),
    )

    membership_df = build_hot_subset_membership_summary_frame(bundle)
    quality_summary_df = build_hot_subset_quality_summary_frame(bundle.hot_subset_source_df)
    quality_reason_df = build_hot_subset_quality_reason_frame(bundle.hot_subset_source_df)
    scored_prediction_df = build_hot_subset_scored_prediction_frame(
        bundle.scored_pass_hot_subset_df
    )
    non_ob_preview_df = build_hot_subset_high_confidence_non_ob_preview_frame(
        bundle.scored_pass_hot_subset_df
    )
    predicted_physics_df = build_hot_subset_predicted_physics_summary_frame(
        bundle.scored_pass_hot_subset_df
    )
    final_outcome_df = build_hot_subset_final_outcome_distribution_frame(
        bundle.final_hot_subset_df
    )
    final_coarse_df = build_hot_subset_final_coarse_distribution_frame(
        bundle.final_hot_subset_df
    )
    final_reason_df = build_hot_subset_final_reason_frame(bundle.final_hot_subset_df)

    assert _require_int_scalar(membership_df.loc[0, "n_rows_true_o_source"]) == 5
    assert _require_int_scalar(membership_df.loc[0, "n_rows_hot_subset"]) == 3
    assert _require_int_scalar(membership_df.loc[0, "n_rows_hot_pass"]) == 2
    assert "pass" in set(quality_summary_df["quality_state"])
    assert "high_ruwe" in set(quality_reason_df["quality_reason"])
    assert int(bundle.pass_hot_subset_df.shape[0]) == 2
    assert int(bundle.scored_pass_hot_subset_df.shape[0]) == 2
    assert "coarse_predicted_label" in bundle.scored_pass_hot_subset_df.columns
    assert _require_int_scalar(scored_prediction_df["n_rows"].sum()) == 2
    assert int(non_ob_preview_df.shape[0]) <= 2
    assert "median_teff_gspphot" in predicted_physics_df.columns
    assert "id" in set(final_outcome_df["final_domain_state"])
    assert "O" in set(final_coarse_df["final_coarse_class"])
    assert "quality_unknown" in set(final_reason_df["final_decision_reason"])
