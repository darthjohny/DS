# Review-хелперы для узкой границы `O/B` в coarse classifier.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from exohost.datasets.archive_research.load_coarse_ob_boundary_review_dataset import (
    load_coarse_ob_boundary_review_dataset,
)
from exohost.db.engine import make_read_only_engine
from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.models.inference import build_probability_frame, require_feature_columns
from exohost.models.protocol import ClassifierModel
from exohost.posthoc.probability_summary import build_probability_summary_frame
from exohost.reporting.model_artifacts import load_model_artifact


@dataclass(frozen=True, slots=True)
class CoarseOBBoundaryReviewConfig:
    # Project baseline: narrow hot pass-slice around the `O/B` boundary.
    quality_state: str = "pass"
    teff_min_k: float = 10_000.0


@dataclass(frozen=True, slots=True)
class CoarseOBBoundaryReviewBundle:
    # Полный пакет review для narrow `O/B` boundary source.
    config: CoarseOBBoundaryReviewConfig
    source_df: pd.DataFrame
    scored_df: pd.DataFrame


_DEFAULT_BOUNDARY_CONFIG = CoarseOBBoundaryReviewConfig()


def load_coarse_ob_boundary_review_bundle_from_env(
    *,
    coarse_model_run_dir: str | Path,
    config: CoarseOBBoundaryReviewConfig = _DEFAULT_BOUNDARY_CONFIG,
    source_limit: int | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> CoarseOBBoundaryReviewBundle:
    # Загружаем live boundary source из БД и связываем его с coarse artifact.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        source_df = load_coarse_ob_boundary_review_dataset(
            engine,
            quality_state=config.quality_state,
            teff_min_k=config.teff_min_k,
            limit=source_limit,
        )
    finally:
        engine.dispose()
    return build_coarse_ob_boundary_review_bundle(
        source_df,
        coarse_model_run_dir=coarse_model_run_dir,
        config=config,
    )


def build_coarse_ob_boundary_review_bundle(
    source_df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
    config: CoarseOBBoundaryReviewConfig = _DEFAULT_BOUNDARY_CONFIG,
) -> CoarseOBBoundaryReviewBundle:
    # Собираем review bundle для narrow `O/B` boundary source.
    scored_df = build_coarse_ob_boundary_scored_frame(
        source_df,
        coarse_model_run_dir=coarse_model_run_dir,
    )
    return CoarseOBBoundaryReviewBundle(
        config=config,
        source_df=source_df,
        scored_df=scored_df,
    )


def build_boundary_membership_summary_frame(
    bundle: CoarseOBBoundaryReviewBundle,
) -> pd.DataFrame:
    # Компактная сводка по baseline boundary-slice.
    rows = [
        {
            "quality_state": bundle.config.quality_state,
            "teff_min_k": float(bundle.config.teff_min_k),
            "n_rows_boundary_source": int(bundle.source_df.shape[0]),
            "n_rows_scored": int(bundle.scored_df.shape[0]),
        }
    ]
    return pd.DataFrame(rows)


def build_boundary_true_class_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Сколько true `O` и true `B` строк есть в narrow boundary source.
    return _build_distribution_frame(
        df,
        column_name="spectral_class",
        label_name="true_spectral_class",
    )


def build_boundary_predicted_class_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Во что coarse-model переводит строки narrow boundary source.
    return _build_distribution_frame(
        df,
        column_name="coarse_predicted_label",
        label_name="coarse_predicted_label",
    )


def build_boundary_confusion_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Confusion-like таблица только для narrow `O/B` source.
    if df.empty:
        return pd.DataFrame(
            columns=["true_spectral_class", "coarse_predicted_label", "n_rows", "share_within_true_class"]
        )

    grouped_df = (
        df.groupby(["spectral_class", "coarse_predicted_label"], dropna=False, sort=True)
        .agg(n_rows=("source_id", "size"))
        .reset_index()
        .rename(columns={"spectral_class": "true_spectral_class"})
    )
    total_by_true_class = (
        grouped_df.groupby("true_spectral_class", dropna=False, sort=True)["n_rows"]
        .transform("sum")
        .astype("int64")
    )
    grouped_df["share_within_true_class"] = grouped_df["n_rows"] / total_by_true_class
    return grouped_df.sort_values(
        ["true_spectral_class", "n_rows", "coarse_predicted_label"],
        ascending=[True, False, True],
        kind="mergesort",
        ignore_index=True,
    )


def build_boundary_probability_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Сравниваем `P(O)` и `P(B)` на true `O` и true `B`.
    required_columns = {"spectral_class", "coarse_probability__O", "coarse_probability__B"}
    if not required_columns.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "true_spectral_class",
                "n_rows",
                "median_probability__O",
                "median_probability__B",
                "mean_probability__O",
                "mean_probability__B",
            ]
        )
    grouped_df = (
        df.groupby("spectral_class", dropna=False, sort=True)
        .agg(
            n_rows=("source_id", "size"),
            median_probability__O=("coarse_probability__O", "median"),
            median_probability__B=("coarse_probability__B", "median"),
            mean_probability__O=("coarse_probability__O", "mean"),
            mean_probability__B=("coarse_probability__B", "mean"),
        )
        .reset_index()
        .rename(columns={"spectral_class": "true_spectral_class"})
    )
    return grouped_df.sort_values(
        ["true_spectral_class"],
        ascending=[True],
        kind="mergesort",
        ignore_index=True,
    )


def build_boundary_physics_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Сравниваем физические параметры true `O` и true `B` на narrow boundary source.
    if df.empty:
        return pd.DataFrame(
            columns=[
                "true_spectral_class",
                "n_rows",
                "median_teff_gspphot",
                "median_logg_gspphot",
                "median_bp_rp",
                "median_radius_flame",
            ]
        )
    grouped_df = (
        df.groupby("spectral_class", dropna=False, sort=True)
        .agg(
            n_rows=("source_id", "size"),
            median_teff_gspphot=("teff_gspphot", "median"),
            median_logg_gspphot=("logg_gspphot", "median"),
            median_bp_rp=("bp_rp", "median"),
            median_radius_flame=("radius_flame", "median"),
        )
        .reset_index()
        .rename(columns={"spectral_class": "true_spectral_class"})
    )
    return grouped_df.sort_values(
        ["true_spectral_class"],
        ascending=[True],
        kind="mergesort",
        ignore_index=True,
    )


def build_high_confidence_o_to_b_preview_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Самые уверенные случаи true `O`, которые coarse уводит в `B`.
    return _build_high_confidence_preview_frame(
        df,
        true_class="O",
        predicted_class="B",
        top_n=top_n,
    )


def build_high_confidence_b_to_o_preview_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Самые уверенные случаи true `B`, которые coarse уводит в `O`.
    return _build_high_confidence_preview_frame(
        df,
        true_class="B",
        predicted_class="O",
        top_n=top_n,
    )


def build_coarse_ob_boundary_scored_frame(
    df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
) -> pd.DataFrame:
    # Считаем coarse predictions и полные `P(O)/P(B)` для narrow boundary source.
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
        raise ValueError("Coarse boundary review requires predict_proba support.")

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
            "spectral_class",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_flame",
        )
        if name in scoring_frame.columns
    ]
    result = scoring_frame.loc[:, base_columns].copy()
    result = result.join(renamed_probability_frame).join(summary_frame)
    result["coarse_model_name"] = artifact.model_name
    return result


def _build_high_confidence_preview_frame(
    df: pd.DataFrame,
    *,
    true_class: str,
    predicted_class: str,
    top_n: int,
) -> pd.DataFrame:
    preview_columns = [
        name
        for name in (
            "source_id",
            "spectral_class",
            "coarse_predicted_label",
            "coarse_probability__O",
            "coarse_probability__B",
            "coarse_probability_max",
            "coarse_probability_margin",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_flame",
        )
        if name in df.columns
    ]
    filtered_df = df.loc[
        (df["spectral_class"].astype("string").str.upper() == true_class)
        & (df["coarse_predicted_label"].astype("string").str.upper() == predicted_class),
        preview_columns,
    ].copy()
    return filtered_df.sort_values(
        ["coarse_probability_max", "coarse_probability_margin"],
        ascending=[False, False],
        kind="mergesort",
        ignore_index=True,
    ).head(top_n)


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
            "Coarse O/B boundary review requires a coarse model artifact for task "
            f"{GAIA_ID_COARSE_CLASSIFICATION_TASK.name}."
        )
    if target_column != GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column:
        raise ValueError(
            "Coarse O/B boundary review requires target_column "
            f"{GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column}."
        )


def _build_distribution_frame(
    df: pd.DataFrame,
    *,
    column_name: str,
    label_name: str,
) -> pd.DataFrame:
    if column_name not in df.columns:
        return pd.DataFrame(columns=[label_name, "n_rows", "share"])
    value_counts = df[column_name].astype("string").fillna("<NA>").value_counts(dropna=False)
    total_rows = int(value_counts.sum())
    rows: list[dict[str, object]] = []
    for label_value, n_rows in value_counts.items():
        rows.append(
            {
                label_name: str(label_value),
                "n_rows": int(n_rows),
                "share": float(n_rows / total_rows) if total_rows > 0 else 0.0,
            }
        )
    return pd.DataFrame.from_records(rows)
