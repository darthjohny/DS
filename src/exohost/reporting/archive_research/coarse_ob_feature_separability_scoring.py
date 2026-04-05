# Model-scoring слой для review feature separability `O vs B`.

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.models.inference import build_probability_frame, require_feature_columns
from exohost.models.protocol import ClassifierModel
from exohost.posthoc.probability_summary import build_probability_summary_frame
from exohost.reporting.archive_research.coarse_ob_feature_separability_contracts import (
    DEFAULT_COARSE_OB_SEPARABILITY_CONFIG,
    CoarseOBFeatureSeparabilityConfig,
)
from exohost.reporting.model_artifacts import load_model_artifact


def build_boundary_permutation_importance_frame(
    df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
    config: CoarseOBFeatureSeparabilityConfig = DEFAULT_COARSE_OB_SEPARABILITY_CONFIG,
) -> pd.DataFrame:
    # Считаем permutation importance текущей coarse-модели на narrow boundary source.
    artifact = load_model_artifact(coarse_model_run_dir)
    _require_coarse_artifact(artifact)
    scoring_frame = _build_boundary_scoring_frame(df)
    require_feature_columns(scoring_frame, feature_columns=artifact.feature_columns)
    model = cast(ClassifierModel, artifact.estimator)

    importance_result = permutation_importance(
        model,
        scoring_frame.loc[:, list(artifact.feature_columns)].copy(),
        scoring_frame.loc[:, "spec_class"].astype(str),
        n_repeats=config.permutation_n_repeats,
        random_state=config.permutation_random_state,
        n_jobs=1,
        scoring={
            "accuracy": _binary_ob_accuracy_scorer,
            "balanced_accuracy": _binary_ob_balanced_accuracy_scorer,
        },
    )

    rows: list[dict[str, object]] = []
    for metric_name, bunch in importance_result.items():
        for feature_name, mean_value, std_value in zip(
            artifact.feature_columns,
            bunch.importances_mean,
            bunch.importances_std,
            strict=True,
        ):
            rows.append(
                {
                    "metric_name": metric_name,
                    "feature_name": feature_name,
                    "importance_mean": float(mean_value),
                    "importance_std": float(std_value),
                }
            )

    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["metric_name", "importance_mean", "feature_name"],
        ascending=[True, False, True],
        kind="mergesort",
        ignore_index=True,
    )


def build_coarse_ob_boundary_scored_frame(
    df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
) -> pd.DataFrame:
    # Считаем coarse predictions и полные `P(O)/P(B)` для train-time `O/B` boundary.
    artifact = load_model_artifact(coarse_model_run_dir)
    _require_coarse_artifact(artifact)
    scoring_frame = _build_boundary_scoring_frame(df)
    model = cast(ClassifierModel, artifact.estimator)
    require_feature_columns(scoring_frame, feature_columns=artifact.feature_columns)
    probability_frame = build_probability_frame(
        model,
        scoring_frame.loc[:, list(artifact.feature_columns)].copy(),
    )
    if probability_frame is None:
        raise ValueError("Coarse O/B separability review requires predict_proba support.")

    summary_frame = build_probability_summary_frame(
        probability_frame,
        prediction_column_name="coarse_predicted_label",
        confidence_column_name="coarse_probability_max",
        margin_column_name="coarse_probability_margin",
    )
    probability_columns = {
        column_name: f"coarse_probability__{column_name}"
        for column_name in probability_frame.columns.astype(str).tolist()
    }
    renamed_probability_frame = probability_frame.rename(columns=probability_columns)
    base_columns = [
        name
        for name in (
            "source_id",
            "spec_class",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_feature",
        )
        if name in scoring_frame.columns
    ]
    result = scoring_frame.loc[:, base_columns].copy()
    result = result.join(renamed_probability_frame).join(summary_frame)
    result["coarse_model_name"] = artifact.model_name
    return result


def _build_boundary_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "radius_feature" not in result.columns and "radius_flame" in result.columns:
        result["radius_feature"] = result["radius_flame"]
    return result


def _require_coarse_artifact(artifact: object) -> None:
    task_name = getattr(artifact, "task_name", None)
    target_column = getattr(artifact, "target_column", None)
    if task_name != GAIA_ID_COARSE_CLASSIFICATION_TASK.name:
        raise ValueError(
            "Coarse O/B feature separability review requires a coarse model artifact "
            f"for task {GAIA_ID_COARSE_CLASSIFICATION_TASK.name}."
        )
    if target_column != GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column:
        raise ValueError(
            "Coarse O/B feature separability review requires target_column "
            f"{GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column}."
        )


def _binary_ob_accuracy_scorer(
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    y_true, y_pred = _build_binary_ob_labels(estimator, X, y)
    return float(accuracy_score(y_true, y_pred))


def _binary_ob_balanced_accuracy_scorer(
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    y_true, y_pred = _build_binary_ob_labels(estimator, X, y)
    return float(balanced_accuracy_score(y_true, y_pred))


def _build_binary_ob_labels(
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    model = cast(ClassifierModel, estimator)
    predicted_labels = model.predict(X)
    y_true = y.astype(str).to_numpy() == "O"
    y_pred = predicted_labels.astype(str) == "O"
    return y_true, y_pred
