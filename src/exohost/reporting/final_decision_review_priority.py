# Файл `final_decision_review_priority.py` слоя `reporting`.
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
    TOP_PRIORITY_COLUMNS,
    FinalDecisionReviewBundle,
)
from exohost.reporting.final_decision_review_distributions import build_distribution_frame


def build_priority_distribution_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Распределение final priority по ranking output.
    if bundle.priority_ranking_df.empty or "priority_label" not in bundle.priority_ranking_df.columns:
        return pd.DataFrame(columns=["priority_label", "n_rows", "share"])
    return build_distribution_frame(
        bundle.priority_ranking_df,
        column_name="priority_label",
        label_name="priority_label",
    )


def build_priority_reason_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Возвращаем top ranking reasons для explainable priority review.
    distribution_df = build_distribution_frame(
        bundle.priority_ranking_df,
        column_name="priority_reason",
        label_name="priority_reason",
    )
    return distribution_df.head(top_n).copy()


def build_priority_component_quantiles_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    quantiles: tuple[float, ...] = (
        0.0,
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        0.99,
        1.0,
    ),
    component_columns: tuple[str, ...] = (
        "priority_score",
        "host_similarity_score",
        "observability_score",
        "class_priority_score",
        "brightness_score",
        "distance_score",
        "astrometry_score",
    ),
) -> pd.DataFrame:
    # Показываем квантильный профиль ranking-компонент для поиска saturation.
    ranking_df = _build_numeric_priority_component_frame(
        bundle.priority_ranking_df,
        component_columns=component_columns,
    )
    if ranking_df.empty:
        return pd.DataFrame(columns=["quantile", *component_columns])

    quantile_frame = ranking_df.quantile(list(quantiles)).reset_index()
    quantile_frame = quantile_frame.rename(columns={"index": "quantile"})
    quantile_frame["quantile"] = quantile_frame["quantile"].astype(float)
    return quantile_frame.copy()


def build_priority_by_coarse_class_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Смотрим, не насыщается ли priority внутри отдельных coarse-классов.
    if bundle.priority_ranking_df.empty:
        return pd.DataFrame(
            columns=[
                "final_coarse_class",
                "n_rows",
                "mean_priority_score",
                "median_priority_score",
                "max_priority_score",
                "min_priority_score",
            ]
        )

    merged_df = bundle.priority_ranking_df.merge(
        bundle.final_decision_df.loc[:, ["source_id", "final_coarse_class"]],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    merged_df["priority_score"] = pd.to_numeric(
        merged_df["priority_score"],
        errors="coerce",
    )
    merged_df = merged_df.dropna(subset=["priority_score"])
    if merged_df.empty:
        return pd.DataFrame(
            columns=[
                "final_coarse_class",
                "n_rows",
                "mean_priority_score",
                "median_priority_score",
                "max_priority_score",
                "min_priority_score",
            ]
        )

    grouped_df = (
        merged_df.groupby("final_coarse_class", dropna=False, sort=True)["priority_score"]
        .agg(["count", "mean", "median", "max", "min"])
        .reset_index()
        .rename(
            columns={
                "count": "n_rows",
                "mean": "mean_priority_score",
                "median": "median_priority_score",
                "max": "max_priority_score",
                "min": "min_priority_score",
            }
        )
        .sort_values("mean_priority_score", ascending=False, kind="mergesort", ignore_index=True)
    )
    grouped_df["final_coarse_class"] = grouped_df["final_coarse_class"].astype(str)
    return grouped_df


def build_host_priority_status_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Явно показываем, подключены ли host/priority сигналы в текущем run.
    has_host_signal = _has_host_signal(bundle)
    has_priority_output = bool(bundle.priority_ranking_df.shape[0] > 0)

    if has_host_signal and has_priority_output:
        note = "В этом прогоне подключены и host-сигнал, и итоговый priority."
    elif has_host_signal and not has_priority_output:
        note = (
            "Host-сигнал есть во входных данных final decision, "
            "но ranking-output пуст. Нужно проверить priority input и пороги."
        )
    else:
        note = (
            "Host/priority интеграция не подключена в текущем чистом контракте. "
            "Поля priority остаются пустыми по дизайну этого прогона."
        )

    return pd.DataFrame(
        [
            {
                "host_signal_available": has_host_signal,
                "priority_input_rows": int(bundle.priority_input_df.shape[0]),
                "priority_output_rows": int(bundle.priority_ranking_df.shape[0]),
                "status_note": note,
            }
        ]
    )


def _has_host_signal(bundle: FinalDecisionReviewBundle) -> bool:
    signal_column_name = "host_similarity_score"
    return any(
        signal_column_name in frame.columns
        for frame in (
            bundle.decision_input_df,
            bundle.priority_input_df,
            bundle.priority_ranking_df,
            bundle.final_decision_df,
        )
    )


def build_top_priority_candidates_frame(
    bundle: FinalDecisionReviewBundle,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Top-N приоритетных объектов с подмешанным final routing context.
    if bundle.priority_ranking_df.empty:
        return pd.DataFrame(columns=TOP_PRIORITY_COLUMNS)

    merged_df = bundle.priority_ranking_df.merge(
        bundle.final_decision_df.loc[
            :,
            [
                name
                for name in (
                    "source_id",
                    "final_domain_state",
                    "final_quality_state",
                    "final_coarse_class",
                    "final_refinement_label",
                )
                if name in bundle.final_decision_df.columns
            ],
        ],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    preview_columns = [name for name in TOP_PRIORITY_COLUMNS if name in merged_df.columns]
    return merged_df.loc[:, preview_columns].head(top_n).copy()


def _build_numeric_priority_component_frame(
    df: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
) -> pd.DataFrame:
    available_columns = [column_name for column_name in component_columns if column_name in df.columns]
    if not available_columns:
        return pd.DataFrame(columns=component_columns)
    numeric_df = df.loc[:, available_columns].copy()
    for column_name in available_columns:
        numeric_df[column_name] = pd.to_numeric(numeric_df[column_name], errors="coerce")
    numeric_df = numeric_df.dropna(how="all")
    return numeric_df


__all__ = [
    "build_host_priority_status_frame",
    "build_priority_by_coarse_class_frame",
    "build_priority_component_quantiles_frame",
    "build_priority_distribution_frame",
    "build_priority_reason_frame",
    "build_top_priority_candidates_frame",
]
