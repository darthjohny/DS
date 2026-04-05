# Файл `priority_threshold_review_frames.py` слоя `reporting`.
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

from exohost.ranking.priority_score import PriorityThresholds
from exohost.reporting.priority_threshold_review_contracts import (
    DEFAULT_PRIORITY_THRESHOLD_VARIANTS,
    PriorityThresholdVariant,
)


def build_priority_threshold_variant_summary_frame(
    df: pd.DataFrame,
    *,
    variants: tuple[PriorityThresholdVariant, ...] = DEFAULT_PRIORITY_THRESHOLD_VARIANTS,
    score_column: str = "priority_score",
    baseline_label_column: str = "priority_label",
) -> pd.DataFrame:
    # Сравниваем распределения low/medium/high между threshold variants.
    score_series = _require_numeric_score_series(df, score_column=score_column)
    n_rows = int(score_series.notna().sum())
    rows: list[dict[str, object]] = []

    for variant in variants:
        label_series = build_priority_label_series(
            score_series,
            thresholds=variant.thresholds,
        )
        counts = label_series.value_counts(dropna=False)
        n_high = _count_label(counts, "high")
        n_medium = _count_label(counts, "medium")
        n_low = _count_label(counts, "low")
        row: dict[str, object] = {
            "variant_name": variant.name,
            "high_min": float(variant.thresholds.high_min),
            "medium_min": float(variant.thresholds.medium_min),
            "n_rows": n_rows,
            "n_high": n_high,
            "share_high": _safe_share(n_high, n_rows),
            "n_medium": n_medium,
            "share_medium": _safe_share(n_medium, n_rows),
            "n_low": n_low,
            "share_low": _safe_share(n_low, n_rows),
        }
        if baseline_label_column in df.columns:
            baseline_label = _require_label_series(df, column_name=baseline_label_column)
            changed_mask = baseline_label.astype("string").ne(label_series)
            n_changed = int(changed_mask.sum())
            row["n_changed_from_baseline"] = n_changed
            row["share_changed_from_baseline"] = _safe_share(
                n_changed,
                n_rows,
            )
        rows.append(row)

    return pd.DataFrame.from_records(rows)


def build_priority_label_transition_frame(
    df: pd.DataFrame,
    *,
    variant: PriorityThresholdVariant,
    score_column: str = "priority_score",
    baseline_label_column: str = "priority_label",
) -> pd.DataFrame:
    # Показываем, как baseline labels переходят в variant labels.
    if baseline_label_column not in df.columns:
        return pd.DataFrame(columns=["baseline_label", "variant_label", "n_rows", "share"])

    score_series = _require_numeric_score_series(df, score_column=score_column)
    baseline_label = _require_label_series(df, column_name=baseline_label_column)
    variant_label = build_priority_label_series(
        score_series,
        thresholds=variant.thresholds,
    )
    review_df = pd.DataFrame(
        {
            "baseline_label": baseline_label.astype("string"),
            "variant_label": variant_label,
        }
    )
    total_rows = int(review_df.shape[0])
    transition_df = review_df.value_counts(
        subset=["baseline_label", "variant_label"],
        dropna=False,
        sort=True,
    ).rename("n_rows").reset_index()
    transition_df = transition_df.sort_values(
        ["n_rows", "baseline_label", "variant_label"],
        ascending=[False, True, True],
        kind="mergesort",
        ignore_index=True,
    )
    transition_df["share"] = transition_df["n_rows"].astype(float) / float(total_rows)
    return transition_df


def build_priority_variant_by_class_frame(
    df: pd.DataFrame,
    *,
    variant: PriorityThresholdVariant,
    score_column: str = "priority_score",
    class_column: str = "final_coarse_class",
) -> pd.DataFrame:
    # Смотрим, как threshold variant влияет на coarse-class groups.
    if class_column not in df.columns:
        return pd.DataFrame(
            columns=[
                class_column,
                "n_rows",
                "n_high",
                "share_high",
                "n_medium",
                "share_medium",
                "n_low",
                "share_low",
            ]
        )

    score_series = _require_numeric_score_series(df, score_column=score_column)
    class_series = _require_label_series(df, column_name=class_column)
    variant_label = build_priority_label_series(
        score_series,
        thresholds=variant.thresholds,
    )
    review_df = pd.DataFrame(
        {
            class_column: class_series.astype("string"),
            "variant_label": variant_label,
        }
    )

    rows: list[dict[str, object]] = []
    for class_value, group_df in review_df.groupby(class_column, dropna=False, sort=True):
        counts = group_df["variant_label"].value_counts(dropna=False)
        n_rows = int(group_df.shape[0])
        n_high = _count_label(counts, "high")
        n_medium = _count_label(counts, "medium")
        n_low = _count_label(counts, "low")
        rows.append(
            {
                class_column: str(class_value),
                "n_rows": n_rows,
                "n_high": n_high,
                "share_high": _safe_share(n_high, n_rows),
                "n_medium": n_medium,
                "share_medium": _safe_share(n_medium, n_rows),
                "n_low": n_low,
                "share_low": _safe_share(n_low, n_rows),
            }
        )

    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    return result.sort_values("n_rows", ascending=False, kind="mergesort", ignore_index=True)


def build_priority_label_series(
    score_series: pd.Series,
    *,
    thresholds: PriorityThresholds,
) -> pd.Series:
    # Векторизованно пересчитываем low/medium/high по выбранным порогам.
    numeric_score = pd.to_numeric(score_series, errors="coerce")
    if not isinstance(numeric_score, pd.Series):
        raise TypeError("priority_score must resolve to a pandas Series.")

    label_series = pd.Series(pd.NA, index=numeric_score.index, dtype="string")
    valid_mask = numeric_score.notna()
    if not bool(valid_mask.any()):
        return label_series

    label_series.loc[valid_mask] = "low"
    label_series.loc[valid_mask & numeric_score.ge(thresholds.medium_min)] = "medium"
    label_series.loc[valid_mask & numeric_score.ge(thresholds.high_min)] = "high"
    return label_series


def _require_numeric_score_series(
    df: pd.DataFrame,
    *,
    score_column: str,
) -> pd.Series:
    if score_column not in df.columns:
        raise ValueError(f"priority review frame is missing required score column: {score_column}")
    score_series = pd.to_numeric(df.loc[:, score_column], errors="coerce")
    if not isinstance(score_series, pd.Series):
        raise TypeError("priority_score must resolve to a pandas Series.")
    return score_series


def _require_label_series(df: pd.DataFrame, *, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        raise ValueError(f"priority review frame is missing required column: {column_name}")
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column


def _safe_share(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(value / total)


def _count_label(counts: pd.Series, label_name: str) -> int:
    value = counts.get(label_name, 0)
    if value is None or value is pd.NA:
        return 0
    return int(value)


__all__ = [
    "build_priority_label_series",
    "build_priority_label_transition_frame",
    "build_priority_threshold_variant_summary_frame",
    "build_priority_variant_by_class_frame",
]
