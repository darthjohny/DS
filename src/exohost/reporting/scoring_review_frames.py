# Файл `scoring_review_frames.py` слоя `reporting`.
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

from exohost.contracts.label_contract import (
    LOW_PRIORITY_SPECTRAL_CLASSES,
    TARGET_SPECTRAL_CLASSES,
    normalize_evolution_stage,
)
from exohost.reporting.scoring_review_bundle import (
    require_prediction_column,
    require_ranking_frame,
)
from exohost.reporting.scoring_review_contracts import (
    TOP_CANDIDATE_COLUMNS,
    ScoringReviewBundle,
)


def build_scoring_summary_frame(bundle: ScoringReviewBundle) -> pd.DataFrame:
    # Собираем компактную таблицу верхнего уровня по scoring/ranking прогону.
    scoring_metadata = bundle.scoring_metadata
    context = scoring_metadata.get("context", {})
    return pd.DataFrame(
        [
            {
                "task_name": scoring_metadata.get("task_name", "unknown"),
                "target_column": scoring_metadata.get("target_column", "unknown"),
                "model_name": scoring_metadata.get("model_name", "unknown"),
                "n_rows": scoring_metadata.get("n_rows", int(bundle.scoring_df.shape[0])),
                "created_at_utc": scoring_metadata.get("created_at_utc", "unknown"),
                "score_mode": context.get("score_mode", "unknown")
                if isinstance(context, dict)
                else "unknown",
                "has_ranking": bundle.ranking_df is not None,
            }
        ]
    )


def build_prediction_distribution_frame(bundle: ScoringReviewBundle) -> pd.DataFrame:
    # Собираем распределение предсказаний модели по scored output.
    prediction_column = require_prediction_column(
        bundle.scoring_df,
        target_column=str(bundle.scoring_metadata["target_column"]),
    )
    prediction_counts = (
        bundle.scoring_df.loc[:, prediction_column]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
    )
    total_rows = int(prediction_counts.sum())
    rows: list[dict[str, object]] = []
    for prediction_label, n_rows in prediction_counts.items():
        rows.append(
            {
                "prediction_label": str(prediction_label),
                "n_rows": int(n_rows),
                "share": float(n_rows / total_rows),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_priority_distribution_frame(bundle: ScoringReviewBundle) -> pd.DataFrame:
    # Собираем распределение final-priority по ranking output.
    ranking_df = require_ranking_frame(bundle)
    if "priority_label" not in ranking_df.columns:
        raise ValueError("Ranking review expected priority_label column.")

    label_counts = (
        ranking_df.loc[:, "priority_label"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
    )
    total_rows = int(label_counts.sum())
    rows: list[dict[str, object]] = []
    for priority_label, n_rows in label_counts.items():
        rows.append(
            {
                "priority_label": str(priority_label),
                "n_rows": int(n_rows),
                "share": float(n_rows / total_rows),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_observability_coverage_frame(bundle: ScoringReviewBundle) -> pd.DataFrame:
    # Показываем, насколько ranking реально опирался на наблюдательные сигналы.
    ranking_df = require_ranking_frame(bundle)
    rows: list[dict[str, object]] = []

    for column_name in (
        "brightness_available",
        "distance_available",
        "astrometry_available",
    ):
        if column_name in ranking_df.columns:
            rows.append(
                {
                    "metric_name": column_name,
                    "metric_value": float(ranking_df.loc[:, column_name].astype(bool).mean()),
                }
            )

    if "observability_evidence_count" in ranking_df.columns:
        rows.append(
            {
                "metric_name": "mean_observability_evidence_count",
                "metric_value": float(
                    ranking_df.loc[:, "observability_evidence_count"].astype(float).mean()
                ),
            }
        )

    return pd.DataFrame.from_records(rows)


def build_top_candidates_frame(
    bundle: ScoringReviewBundle,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Возвращаем срез top-N кандидатов по готовому ranking output.
    ranking_df = require_ranking_frame(bundle)
    preview_columns = [name for name in TOP_CANDIDATE_COLUMNS if name in ranking_df.columns]
    return ranking_df.loc[:, preview_columns].head(top_n).copy()


def build_goal_alignment_frame(
    bundle: ScoringReviewBundle,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Проверяем, насколько top-N ranking соответствует задаче follow-up отбора.
    ranking_df = require_ranking_frame(bundle)
    top_frame = ranking_df.head(top_n).copy()
    if top_frame.empty:
        raise ValueError("Ranking review requires at least one row.")

    spec_classes = top_frame.loc[:, "spec_class"].astype(str).str.upper()
    evolution_stages = (
        top_frame.loc[:, "evolution_stage"].astype(str).map(normalize_evolution_stage)
        if "evolution_stage" in top_frame.columns
        else pd.Series(["unknown"] * int(top_frame.shape[0]))
    )
    n_rows = int(top_frame.shape[0])
    rows: list[dict[str, object]] = [
        _metric_row("top_n", n_rows),
        _metric_row(
            "target_class_share",
            float(spec_classes.isin(TARGET_SPECTRAL_CLASSES).mean()),
        ),
        _metric_row(
            "low_priority_class_share",
            float(spec_classes.isin(LOW_PRIORITY_SPECTRAL_CLASSES).mean()),
        ),
        _metric_row(
            "dwarf_share",
            float((evolution_stages == "dwarf").mean()),
        ),
        _metric_row(
            "evolved_share",
            float((evolution_stages == "evolved").mean()),
        ),
    ]

    if "priority_label" in top_frame.columns:
        priority_labels = top_frame.loc[:, "priority_label"].astype(str)
        rows.extend(
            [
                _metric_row("high_priority_share", float((priority_labels == "high").mean())),
                _metric_row("medium_priority_share", float((priority_labels == "medium").mean())),
                _metric_row("low_priority_share", float((priority_labels == "low").mean())),
            ]
        )

    if "host_similarity_score" in top_frame.columns:
        rows.append(
            _metric_row(
                "mean_host_similarity_score",
                float(top_frame.loc[:, "host_similarity_score"].astype(float).mean()),
            )
        )
    if "observability_score" in top_frame.columns:
        rows.append(
            _metric_row(
                "mean_observability_score",
                float(top_frame.loc[:, "observability_score"].astype(float).mean()),
            )
        )
    if "observability_evidence_count" in top_frame.columns:
        rows.append(
            _metric_row(
                "mean_observability_evidence_count",
                float(top_frame.loc[:, "observability_evidence_count"].astype(float).mean()),
            )
        )
    if "priority_score" in top_frame.columns:
        rows.append(
            _metric_row(
                "mean_priority_score",
                float(top_frame.loc[:, "priority_score"].astype(float).mean()),
            )
        )

    return pd.DataFrame.from_records(rows)


def _metric_row(metric_name: str, value: object) -> dict[str, object]:
    # Формируем строку long-format метрики для notebook-представления.
    return {"metric_name": metric_name, "metric_value": value}


__all__ = [
    "build_goal_alignment_frame",
    "build_observability_coverage_frame",
    "build_prediction_distribution_frame",
    "build_priority_distribution_frame",
    "build_scoring_summary_frame",
    "build_top_candidates_frame",
]
