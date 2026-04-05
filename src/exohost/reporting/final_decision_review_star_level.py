# Файл `final_decision_review_star_level.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import pandas as pd

from exohost.reporting.final_decision_review_contracts import (
    STAR_RESULT_COLUMNS,
    FinalDecisionReviewBundle,
)
from exohost.reporting.final_decision_review_distributions import build_distribution_frame


def build_final_coarse_class_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    final_domain_state: str = "id",
    top_n: int = 20,
) -> pd.DataFrame:
    # Распределение coarse-классов для заданного final domain state.
    final_df = bundle.final_decision_df.copy()
    filtered_df = final_df.loc[
        final_df["final_domain_state"].astype(str) == final_domain_state,
        :,
    ]
    distribution_df = build_distribution_frame(
        filtered_df,
        column_name="final_coarse_class",
        label_name="final_coarse_class",
    )
    return distribution_df.head(top_n).copy()


def build_final_refinement_label_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    final_domain_state: str = "id",
    top_n: int = 20,
) -> pd.DataFrame:
    # Распределение финальных refinement labels для заданного final domain state.
    final_df = bundle.final_decision_df.copy()
    filtered_df = final_df.loc[
        final_df["final_domain_state"].astype(str) == final_domain_state,
        :,
    ]
    distribution_df = build_distribution_frame(
        filtered_df,
        column_name="final_refinement_label",
        label_name="final_refinement_label",
    )
    return distribution_df.head(top_n).copy()


def build_star_level_result_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Собираем один merged frame с финальным routing, physics и optional priority fields.
    merged_df = bundle.decision_input_df.merge(
        bundle.final_decision_df,
        on="source_id",
        how="inner",
        validate="one_to_one",
    )
    if not bundle.priority_ranking_df.empty:
        merged_df = merged_df.merge(
            bundle.priority_ranking_df,
            on="source_id",
            how="left",
            validate="one_to_one",
        )

    result = merged_df.copy()
    for column_name in STAR_RESULT_COLUMNS:
        if column_name not in result.columns:
            result[column_name] = pd.NA
    return result.loc[:, list(STAR_RESULT_COLUMNS)].copy()


def build_star_result_preview_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    final_domain_state: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    # Возвращаем compact preview по individual stars для notebook review.
    result = build_star_level_result_frame(bundle)
    if final_domain_state is not None:
        result = result.loc[
            result["final_domain_state"].astype(str) == final_domain_state,
            :,
        ].copy()
    if result.empty:
        return result

    sort_ready = result.copy()
    for column_name in (
        "priority_score",
        "final_refinement_confidence",
        "final_coarse_confidence",
        "parallax_over_error",
    ):
        sort_ready[column_name] = pd.to_numeric(sort_ready[column_name], errors="coerce")

    if final_domain_state == "id":
        sort_ready = sort_ready.sort_values(
            [
                "priority_score",
                "final_refinement_confidence",
                "final_coarse_confidence",
                "parallax_over_error",
                "source_id",
            ],
            ascending=[False, False, False, False, True],
            kind="mergesort",
            ignore_index=True,
            na_position="last",
        )
    else:
        sort_ready = sort_ready.sort_values(
            [
                "final_decision_reason",
                "review_bucket",
                "quality_reason",
                "parallax_over_error",
                "source_id",
            ],
            ascending=[True, True, True, False, True],
            kind="mergesort",
            ignore_index=True,
            na_position="last",
        )

    return sort_ready.head(top_n).copy()


def build_numeric_state_summary_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    group_column: str = "final_domain_state",
    numeric_columns: tuple[str, ...] = (
        "host_similarity_score",
        "parallax",
        "parallax_over_error",
        "ruwe",
        "phot_g_mean_mag",
    ),
) -> pd.DataFrame:
    # Сводка mean/median по числовым признакам в разрезе final state.
    merged_df = bundle.decision_input_df.merge(
        bundle.final_decision_df.loc[:, ["source_id", group_column]],
        on="source_id",
        how="inner",
        validate="one_to_one",
    )

    rows: list[dict[str, object]] = []
    for column_name in numeric_columns:
        if column_name not in merged_df.columns:
            continue
        grouped = merged_df.loc[:, [group_column, column_name]].copy()
        grouped[column_name] = pd.to_numeric(grouped[column_name], errors="coerce")
        grouped = grouped.dropna(subset=[column_name])
        if grouped.empty:
            continue
        for state_value, state_df in grouped.groupby(group_column, dropna=False, sort=True):
            rows.append(
                {
                    "group_value": str(state_value),
                    "metric_name": column_name,
                    "mean_value": float(state_df[column_name].mean()),
                    "median_value": float(state_df[column_name].median()),
                    "n_rows": int(state_df.shape[0]),
                }
            )

    return pd.DataFrame.from_records(rows)


__all__ = [
    "build_final_coarse_class_frame",
    "build_final_refinement_label_frame",
    "build_numeric_state_summary_frame",
    "build_star_level_result_frame",
    "build_star_result_preview_frame",
]
