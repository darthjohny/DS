# Файл `coarse_ob_domain_shift_scoring.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.models.inference import build_probability_frame, require_feature_columns
from exohost.models.protocol import ClassifierModel
from exohost.posthoc.probability_summary import build_probability_summary_frame
from exohost.reporting.model_artifacts import load_model_artifact


def build_coarse_ob_domain_scored_frame(
    df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
    domain_name: str,
) -> pd.DataFrame:
    # Прогоняем current coarse artifact на заданном `O/B` domain source.
    artifact = load_model_artifact(coarse_model_run_dir)
    _require_coarse_artifact(artifact)
    scoring_frame = build_coarse_ob_domain_scoring_frame(df)
    estimator = cast(ClassifierModel, artifact.estimator)
    require_feature_columns(scoring_frame, feature_columns=artifact.feature_columns)
    probability_frame = build_probability_frame(
        estimator,
        scoring_frame.loc[:, list(artifact.feature_columns)].copy(),
    )
    if probability_frame is None:
        raise ValueError("Coarse O/B domain shift review requires predict_proba support.")

    summary_df = build_probability_summary_frame(
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
        column_name
        for column_name in (
            "source_id",
            "spectral_class",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_feature",
            "radius_flame",
            "quality_state",
            "quality_reason",
            "review_bucket",
        )
        if column_name in scoring_frame.columns
    ]
    result = scoring_frame.loc[:, base_columns].copy()
    result = result.join(renamed_probability_frame).join(summary_df)
    result["domain_name"] = domain_name
    result["coarse_model_name"] = artifact.model_name
    return result


def build_coarse_ob_domain_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Нормализуем вход в scoring под coarse feature contract.
    result = df.copy()
    if "spectral_class" not in result.columns and "spec_class" in result.columns:
        result["spectral_class"] = result["spec_class"].astype("string").str.upper()
    if "radius_feature" not in result.columns:
        if "radius_flame" in result.columns:
            result["radius_feature"] = result["radius_flame"]
        elif "radius_gspphot" in result.columns:
            result["radius_feature"] = result["radius_gspphot"]
    return result


def _require_coarse_artifact(artifact: object) -> None:
    task_name = getattr(artifact, "task_name", None)
    target_column = getattr(artifact, "target_column", None)
    if task_name != GAIA_ID_COARSE_CLASSIFICATION_TASK.name:
        raise ValueError(
            "Coarse O/B domain shift review requires a coarse model artifact "
            f"for task {GAIA_ID_COARSE_CLASSIFICATION_TASK.name}."
        )
    if target_column != GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column:
        raise ValueError(
            "Coarse O/B domain shift review requires target_column "
            f"{GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column}."
        )
