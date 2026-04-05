# Review-хелперы для rare-tail анализа true `O` rows в coarse stage.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from exohost.datasets.archive_research.load_coarse_o_review_dataset import (
    load_coarse_o_review_dataset,
)
from exohost.db.engine import make_read_only_engine
from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.posthoc.coarse_scoring import build_coarse_scored_frame
from exohost.reporting.final_decision_review import load_final_decision_review_bundle
from exohost.reporting.model_artifacts import load_model_artifact


@dataclass(frozen=True, slots=True)
class CoarseOReviewBundle:
    # Полный пакет `O` rare-tail review без изменения production pipeline.
    source_df: pd.DataFrame
    pass_source_df: pd.DataFrame
    scored_pass_df: pd.DataFrame
    final_o_df: pd.DataFrame


def load_coarse_o_review_bundle_from_env(
    *,
    coarse_model_run_dir: str | Path,
    final_decision_run_dir: str | Path,
    source_limit: int | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> CoarseOReviewBundle:
    # Загружаем live `O` source из БД и связываем его с coarse artifact и final run.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        source_df = load_coarse_o_review_dataset(engine, limit=source_limit)
    finally:
        engine.dispose()
    return build_coarse_o_review_bundle(
        source_df,
        coarse_model_run_dir=coarse_model_run_dir,
        final_decision_run_dir=final_decision_run_dir,
    )


def build_coarse_o_review_bundle(
    source_df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
    final_decision_run_dir: str | Path,
) -> CoarseOReviewBundle:
    # Собираем единый review bundle для true `O` rows.
    pass_source_df = _build_pass_o_source_frame(source_df)
    scored_pass_df = build_coarse_o_scored_frame(
        pass_source_df,
        coarse_model_run_dir=coarse_model_run_dir,
    )
    final_o_df = build_coarse_o_final_outcome_frame(
        source_df,
        final_decision_run_dir=final_decision_run_dir,
    )
    return CoarseOReviewBundle(
        source_df=source_df,
        pass_source_df=pass_source_df,
        scored_pass_df=scored_pass_df,
        final_o_df=final_o_df,
    )


def build_o_source_quality_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Показываем, сколько true `O` строк живет в каждом quality_state.
    return _build_distribution_frame(
        df,
        column_name="quality_state",
        label_name="quality_state",
    )


def build_o_source_quality_reason_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Возвращаем top quality reasons только для true `O` rows.
    return _build_distribution_frame(
        df,
        column_name="quality_reason",
        label_name="quality_reason",
    ).head(top_n)


def build_o_scored_prediction_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Показываем распределение coarse predictions на pass-части `O`.
    distribution_df = _build_distribution_frame(
        df,
        column_name="coarse_predicted_label",
        label_name="coarse_predicted_label",
    )
    grouped_df = (
        df.groupby("coarse_predicted_label", dropna=False, sort=True)
        .agg(
            mean_confidence=("coarse_probability_max", "mean"),
            median_confidence=("coarse_probability_max", "median"),
            max_confidence=("coarse_probability_max", "max"),
        )
        .reset_index()
    )
    result = distribution_df.merge(
        grouped_df,
        on="coarse_predicted_label",
        how="left",
        validate="one_to_one",
    )
    return result.head(top_n)


def build_o_high_confidence_non_o_preview_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Показываем самые уверенные случаи, где true `O` ушел в другой coarse class.
    preview_columns = [
        name
        for name in (
            "source_id",
            "coarse_predicted_label",
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
        df["coarse_predicted_label"].astype(str).str.upper() != "O",
        preview_columns,
    ].copy()
    return filtered_df.sort_values(
        ["coarse_probability_max", "coarse_probability_margin"],
        ascending=[False, False],
        kind="mergesort",
        ignore_index=True,
    ).head(top_n)


def build_o_predicted_physics_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Смотрим, какими физическими параметрами обладают true `O` rows внутри predicted groups.
    if df.empty:
        return pd.DataFrame(
            columns=[
                "coarse_predicted_label",
                "n_rows",
                "median_teff_gspphot",
                "median_logg_gspphot",
                "median_bp_rp",
                "median_radius_flame",
            ]
        )

    grouped_df = (
        df.groupby("coarse_predicted_label", dropna=False, sort=True)
        .agg(
            n_rows=("source_id", "size"),
            median_teff_gspphot=("teff_gspphot", "median"),
            median_logg_gspphot=("logg_gspphot", "median"),
            median_bp_rp=("bp_rp", "median"),
            median_radius_flame=("radius_flame", "median"),
        )
        .reset_index()
    )
    return grouped_df.sort_values(
        ["n_rows", "coarse_predicted_label"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )


def build_o_final_outcome_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Показываем downstream fate true `O` rows в final decision run.
    return _build_distribution_frame(
        df,
        column_name="final_domain_state",
        label_name="final_domain_state",
    )


def build_o_final_coarse_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Показываем, в какие final coarse classes попали true `O` rows.
    return _build_distribution_frame(
        df,
        column_name="final_coarse_class",
        label_name="final_coarse_class",
    )


def build_o_final_reason_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Показываем top final decision reasons для true `O` rows.
    return _build_distribution_frame(
        df,
        column_name="final_decision_reason",
        label_name="final_decision_reason",
    ).head(top_n)


def build_coarse_o_scored_frame(
    df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
) -> pd.DataFrame:
    # Считаем coarse predictions только для `quality_state = pass` части `O`.
    artifact = load_model_artifact(coarse_model_run_dir)
    _require_coarse_artifact(artifact)
    scoring_frame = _build_coarse_review_scoring_frame(df)
    scored_df = build_coarse_scored_frame(
        scoring_frame,
        estimator=artifact.estimator,
        feature_columns=artifact.feature_columns,
        model_name=artifact.model_name,
    )
    return scoring_frame.merge(
        scored_df,
        on="source_id",
        how="left",
        validate="one_to_one",
    )


def build_coarse_o_final_outcome_frame(
    source_df: pd.DataFrame,
    *,
    final_decision_run_dir: str | Path,
) -> pd.DataFrame:
    # Связываем true `O` source с текущим final decision bundle по `source_id`.
    bundle = load_final_decision_review_bundle(final_decision_run_dir)
    final_columns = [
        "source_id",
        "final_domain_state",
        "final_quality_state",
        "final_coarse_class",
        "final_coarse_confidence",
        "final_refinement_label",
        "final_refinement_state",
        "final_refinement_confidence",
        "final_decision_reason",
    ]
    decision_columns = [
        "source_id",
        "quality_reason",
        "review_bucket",
        "ood_state",
        "ood_reason",
    ]
    merged_df = source_df.merge(
        bundle.final_decision_df.loc[:, final_columns],
        on="source_id",
        how="left",
        validate="one_to_one",
    ).merge(
        bundle.decision_input_df.loc[:, decision_columns],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    return merged_df


def _build_pass_o_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.loc[df["quality_state"].astype(str) == "pass"].copy().reset_index(drop=True)
    )


def _build_coarse_review_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "radius_feature" not in result.columns and "radius_flame" in result.columns:
        result["radius_feature"] = result["radius_flame"]
    return result


def _require_coarse_artifact(artifact: object) -> None:
    task_name = getattr(artifact, "task_name", None)
    target_column = getattr(artifact, "target_column", None)
    if task_name != GAIA_ID_COARSE_CLASSIFICATION_TASK.name:
        raise ValueError(
            "Coarse O review requires a coarse model artifact for task "
            f"{GAIA_ID_COARSE_CLASSIFICATION_TASK.name}."
        )
    if target_column != GAIA_ID_COARSE_CLASSIFICATION_TASK.target_column:
        raise ValueError(
            "Coarse O review requires target_column "
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
