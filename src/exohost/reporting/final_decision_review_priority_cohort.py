# Файл `final_decision_review_priority_cohort.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from typing import cast

import pandas as pd

from exohost.reporting.final_decision_review_contracts import (
    HIGH_PRIORITY_PHYSICS_COLUMNS,
    FinalDecisionReviewBundle,
)
from exohost.reporting.final_decision_review_distributions import build_distribution_frame

HIGH_PRIORITY_COMPONENT_COLUMNS: tuple[str, ...] = (
    "priority_score",
    "host_similarity_score",
    "observability_score",
    "class_priority_score",
    "brightness_score",
    "distance_score",
    "astrometry_score",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "phot_g_mean_mag",
    "radius_flame",
    "lum_flame",
)


def build_high_priority_summary_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Одна строка с размером high-priority слоя и медианами главных сигналов.
    high_df = _build_high_priority_context_frame(bundle)
    if high_df.empty:
        return pd.DataFrame(
            columns=[
                "n_rows",
                "share_priority_ranking",
                "share_final_id",
                "median_priority_score",
                "median_host_similarity_score",
                "median_observability_score",
                "median_parallax_over_error",
                "median_ruwe",
                "median_phot_g_mean_mag",
            ]
        )

    final_id_count = int(
        bundle.final_decision_df.loc[
            bundle.final_decision_df["final_domain_state"].astype(str) == "id",
            :,
        ].shape[0]
    )
    return pd.DataFrame(
        [
            {
                "n_rows": int(high_df.shape[0]),
                "share_priority_ranking": float(high_df.shape[0] / max(bundle.priority_ranking_df.shape[0], 1)),
                "share_final_id": float(high_df.shape[0] / max(final_id_count, 1)),
                "median_priority_score": _series_median(high_df, "priority_score"),
                "median_host_similarity_score": _series_median(high_df, "host_similarity_score"),
                "median_observability_score": _series_median(high_df, "observability_score"),
                "median_parallax_over_error": _series_median(high_df, "parallax_over_error"),
                "median_ruwe": _series_median(high_df, "ruwe"),
                "median_phot_g_mean_mag": _series_median(high_df, "phot_g_mean_mag"),
            }
        ]
    )


def build_high_priority_coarse_class_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    # Распределение coarse-классов внутри high-priority слоя.
    high_df = _build_high_priority_context_frame(bundle)
    distribution_df = build_distribution_frame(
        high_df,
        column_name="final_coarse_class",
        label_name="final_coarse_class",
    )
    return distribution_df.head(top_n).copy()


def build_high_priority_refinement_label_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    # Распределение итоговых refinement labels внутри high-priority слоя.
    high_df = _build_high_priority_context_frame(bundle)
    distribution_df = build_distribution_frame(
        high_df,
        column_name="final_refinement_label",
        label_name="final_refinement_label",
    )
    return distribution_df.head(top_n).copy()


def build_high_priority_component_summary_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Сводка по компонентам score и базовой физике внутри high-priority слоя.
    high_df = _build_high_priority_context_frame(bundle)
    rows: list[dict[str, object]] = []
    for column_name in HIGH_PRIORITY_COMPONENT_COLUMNS:
        numeric_series = _numeric_series(high_df, column_name)
        if numeric_series.empty:
            continue
        rows.append(
            {
                "metric_name": column_name,
                "p25_value": float(numeric_series.quantile(0.25)),
                "median_value": float(numeric_series.median()),
                "p75_value": float(numeric_series.quantile(0.75)),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_high_priority_candidate_physics_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    # Top-N объектов из high-priority слоя с физикой и вкладом score-компонент.
    high_df = _build_high_priority_context_frame(bundle)
    if high_df.empty:
        return pd.DataFrame(columns=HIGH_PRIORITY_PHYSICS_COLUMNS)

    sort_ready = high_df.copy()
    for column_name in (
        "priority_score",
        "host_similarity_score",
        "observability_score",
        "parallax_over_error",
    ):
        sort_ready[column_name] = pd.to_numeric(sort_ready[column_name], errors="coerce")

    sort_ready = sort_ready.sort_values(
        [
            "priority_score",
            "host_similarity_score",
            "observability_score",
            "parallax_over_error",
            "source_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
        ignore_index=True,
        na_position="last",
    )
    preview_columns = [name for name in HIGH_PRIORITY_PHYSICS_COLUMNS if name in sort_ready.columns]
    return sort_ready.loc[:, preview_columns].head(top_n).copy()


def _build_high_priority_context_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    if bundle.priority_ranking_df.empty:
        return pd.DataFrame()

    final_columns = [
        name
        for name in (
            "source_id",
            "final_domain_state",
            "final_quality_state",
            "final_coarse_class",
            "final_refinement_label",
            "final_decision_reason",
        )
        if name in bundle.final_decision_df.columns
    ]
    decision_columns = [
        name
        for name in (
            "source_id",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "phot_g_mean_mag",
            "radius_flame",
            "lum_flame",
        )
        if name in bundle.decision_input_df.columns
    ]

    merged_df = bundle.priority_ranking_df.merge(
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
    return merged_df.loc[merged_df["priority_label"].astype(str) == "high", :].copy()


def _numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(dtype="float64")
    source_series = cast(pd.Series, df.loc[:, column_name])
    numeric_series = cast(
        pd.Series,
        pd.Series(pd.to_numeric(source_series, errors="coerce"), dtype="float64"),
    ).dropna()
    if numeric_series.empty:
        return pd.Series(dtype="float64")
    return numeric_series


def _series_median(df: pd.DataFrame, column_name: str) -> float | None:
    series = _numeric_series(df, column_name)
    if series.empty:
        return None
    return float(series.median())


__all__ = [
    "build_high_priority_candidate_physics_frame",
    "build_high_priority_coarse_class_frame",
    "build_high_priority_component_summary_frame",
    "build_high_priority_refinement_label_frame",
    "build_high_priority_summary_frame",
]
