# Тестовый файл `test_priority_threshold_review.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from numbers import Integral

import pandas as pd

from exohost.ranking.priority_score import PriorityThresholds
from exohost.reporting.priority_threshold_review import (
    PriorityThresholdVariant,
    build_priority_label_series,
    build_priority_label_transition_frame,
    build_priority_threshold_variant_summary_frame,
    build_priority_variant_by_class_frame,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _build_priority_review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "1",
                "priority_score": 0.92,
                "priority_label": "high",
                "final_coarse_class": "G",
            },
            {
                "source_id": "2",
                "priority_score": 0.78,
                "priority_label": "high",
                "final_coarse_class": "K",
            },
            {
                "source_id": "3",
                "priority_score": 0.58,
                "priority_label": "medium",
                "final_coarse_class": "F",
            },
            {
                "source_id": "4",
                "priority_score": 0.31,
                "priority_label": "low",
                "final_coarse_class": "A",
            },
        ]
    )


def test_build_priority_label_series_recomputes_labels() -> None:
    score_series = pd.Series([0.92, 0.78, 0.58, 0.31], dtype="float64")

    label_series = build_priority_label_series(
        score_series,
        thresholds=PriorityThresholds(high_min=0.85, medium_min=0.55),
    )

    assert label_series.tolist() == ["high", "medium", "medium", "low"]


def test_build_priority_threshold_variant_summary_frame_counts_label_changes() -> None:
    review_df = _build_priority_review_frame()
    variant = PriorityThresholdVariant(
        name="strict_high_medium",
        thresholds=PriorityThresholds(high_min=0.85, medium_min=0.55),
    )

    summary_df = build_priority_threshold_variant_summary_frame(
        review_df,
        variants=(variant,),
    )

    assert list(summary_df["variant_name"]) == ["strict_high_medium"]
    assert _require_int_scalar(summary_df.loc[0, "n_high"]) == 1
    assert _require_int_scalar(summary_df.loc[0, "n_medium"]) == 2
    assert _require_int_scalar(summary_df.loc[0, "n_low"]) == 1
    assert _require_int_scalar(summary_df.loc[0, "n_changed_from_baseline"]) == 1


def test_build_priority_label_transition_frame_tracks_baseline_to_variant() -> None:
    review_df = _build_priority_review_frame()
    variant = PriorityThresholdVariant(
        name="strict_high_medium",
        thresholds=PriorityThresholds(high_min=0.85, medium_min=0.55),
    )

    transition_df = build_priority_label_transition_frame(
        review_df,
        variant=variant,
    )

    changed_row = transition_df.loc[
        (transition_df["baseline_label"] == "high")
        & (transition_df["variant_label"] == "medium")
    ]
    assert _require_int_scalar(changed_row.iloc[0]["n_rows"]) == 1


def test_build_priority_variant_by_class_frame_returns_class_level_distribution() -> None:
    review_df = _build_priority_review_frame()
    variant = PriorityThresholdVariant(
        name="strict_high_medium",
        thresholds=PriorityThresholds(high_min=0.85, medium_min=0.55),
    )

    class_df = build_priority_variant_by_class_frame(
        review_df,
        variant=variant,
    )

    assert set(class_df["final_coarse_class"]) == {"A", "F", "G", "K"}
    g_row = class_df.loc[class_df["final_coarse_class"] == "G"].iloc[0]
    k_row = class_df.loc[class_df["final_coarse_class"] == "K"].iloc[0]
    assert _require_int_scalar(g_row["n_high"]) == 1
    assert _require_int_scalar(k_row["n_medium"]) == 1
