"""Агрегация generalization summary для heavy validation слоя."""

from __future__ import annotations

import pandas as pd

from analysis.model_validation.scalars import scalar_to_float, scalar_to_int

CANONICAL_STAGE_ORDER: tuple[str, ...] = (
    "train_in_sample",
    "cv_oof",
    "test_holdout",
)
def build_generalization_stage_frame(repeated_splits_df: pd.DataFrame) -> pd.DataFrame:
    """Нормализовать repeated split diagnostics в train/cv/test stage view."""
    if repeated_splits_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in repeated_splits_df.iterrows():
        base_row = {
            "validation_protocol_name": str(row["validation_protocol_name"]),
            "benchmark_protocol_name": str(row["benchmark_protocol_name"]),
            "split_random_state": scalar_to_int(row["split_random_state"]),
            "model_name": str(row["model_name"]),
            "metric_name": str(row["metric_name"]),
            "metric_direction": str(row["metric_direction"]),
            "precision_k": scalar_to_int(row["precision_k"]),
        }
        rows.append(
            {
                **base_row,
                "stage_name": "train_in_sample",
                "stage_scope": str(row["train_scope"]),
                "stage_value": scalar_to_float(row["train_value"]),
            }
        )
        rows.append(
            {
                **base_row,
                "stage_name": "test_holdout",
                "stage_scope": str(row["test_scope"]),
                "stage_value": scalar_to_float(row["test_value"]),
            }
        )
        cv_score_mean = row["cv_score_mean"]
        if pd.notna(cv_score_mean):
            rows.append(
                {
                    **base_row,
                    "stage_name": "cv_oof",
                    "stage_scope": (
                        str(row["cv_summary_scope"])
                        if row["cv_summary_scope"] is not None
                        else "cv_oof"
                    ),
                    "stage_value": scalar_to_float(cv_score_mean),
                }
            )

    stage_df = pd.DataFrame.from_records(rows)
    stage_order = {name: index for index, name in enumerate(CANONICAL_STAGE_ORDER)}
    return (
        stage_df.assign(
            _stage_order=stage_df["stage_name"].astype(str).map(stage_order),
        )
        .sort_values(
            ["model_name", "metric_name", "_stage_order", "split_random_state"],
            ignore_index=True,
        )
        .drop(columns="_stage_order")
    )


def build_generalization_summary_frame(stage_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегировать train/cv/test stages по repeated split run."""
    if stage_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_columns = [
        "validation_protocol_name",
        "benchmark_protocol_name",
        "model_name",
        "metric_name",
        "stage_name",
        "stage_scope",
    ]
    for (
        validation_protocol_name,
        benchmark_protocol_name,
        model_name,
        metric_name,
        stage_name,
        stage_scope,
    ), group in stage_df.groupby(group_columns, sort=True):
        metric_direction = str(group["metric_direction"].iloc[0])
        stage_values = group["stage_value"].astype(float)
        if metric_direction == "minimize":
            best_idx = stage_values.idxmin()
            worst_idx = stage_values.idxmax()
        else:
            best_idx = stage_values.idxmax()
            worst_idx = stage_values.idxmin()

        rows.append(
            {
                "validation_protocol_name": str(validation_protocol_name),
                "benchmark_protocol_name": str(benchmark_protocol_name),
                "model_name": str(model_name),
                "metric_name": str(metric_name),
                "metric_direction": metric_direction,
                "precision_k": scalar_to_int(group["precision_k"].iloc[0]),
                "stage_name": str(stage_name),
                "stage_scope": str(stage_scope),
                "split_count": int(group.shape[0]),
                "score_mean": float(stage_values.mean()),
                "score_std": float(stage_values.std(ddof=0)),
                "score_min": float(stage_values.min()),
                "score_max": float(stage_values.max()),
                "best_split_random_state": scalar_to_int(
                    group.loc[best_idx, "split_random_state"]
                ),
                "best_score_value": scalar_to_float(group.loc[best_idx, "stage_value"]),
                "worst_split_random_state": scalar_to_int(
                    group.loc[worst_idx, "split_random_state"]
                ),
                "worst_score_value": scalar_to_float(
                    group.loc[worst_idx, "stage_value"]
                ),
            }
        )

    stage_order = {name: index for index, name in enumerate(CANONICAL_STAGE_ORDER)}
    summary_df = pd.DataFrame.from_records(rows)
    return (
        summary_df.assign(
            _stage_order=summary_df["stage_name"].astype(str).map(stage_order),
        )
        .sort_values(
            ["model_name", "metric_name", "_stage_order"],
            ignore_index=True,
        )
        .drop(columns="_stage_order")
    )


def build_gap_diagnostics_frame(repeated_splits_df: pd.DataFrame) -> pd.DataFrame:
    """Собрать агрегированную gap-диагностику между stage-ами."""
    if repeated_splits_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_columns = ["validation_protocol_name", "benchmark_protocol_name", "model_name", "metric_name"]
    for (
        validation_protocol_name,
        benchmark_protocol_name,
        model_name,
        metric_name,
    ), group in repeated_splits_df.groupby(group_columns, sort=True):
        train_test_gaps = group["train_minus_test"].astype(float)
        abs_train_test_gaps = group["abs_train_test_gap"].astype(float)
        cv_test_gaps = group["cv_minus_test"].astype(float)
        valid_cv_test_gaps = cv_test_gaps[~cv_test_gaps.isna()]
        cv_available = not valid_cv_test_gaps.empty

        rows.append(
            {
                "validation_protocol_name": str(validation_protocol_name),
                "benchmark_protocol_name": str(benchmark_protocol_name),
                "model_name": str(model_name),
                "metric_name": str(metric_name),
                "metric_direction": str(group["metric_direction"].iloc[0]),
                "precision_k": scalar_to_int(group["precision_k"].iloc[0]),
                "train_scope": str(group["train_scope"].iloc[0]),
                "test_scope": str(group["test_scope"].iloc[0]),
                "split_count": int(group.shape[0]),
                "refit_split_count": int(group["is_refit_metric"].astype(bool).sum()),
                "train_minus_test_mean": float(train_test_gaps.mean()),
                "train_minus_test_std": float(train_test_gaps.std(ddof=0)),
                "train_minus_test_min": float(train_test_gaps.min()),
                "train_minus_test_max": float(train_test_gaps.max()),
                "abs_train_test_gap_mean": float(abs_train_test_gaps.mean()),
                "abs_train_test_gap_max": float(abs_train_test_gaps.max()),
                "cv_available_splits": int(valid_cv_test_gaps.shape[0]),
                "cv_scope": (
                    str(group.loc[group["cv_summary_scope"].notna(), "cv_summary_scope"].iloc[0])
                    if cv_available
                    else None
                ),
                "cv_minus_test_mean": (
                    float(valid_cv_test_gaps.mean()) if cv_available else float("nan")
                ),
                "cv_minus_test_std": (
                    float(valid_cv_test_gaps.std(ddof=0))
                    if cv_available
                    else float("nan")
                ),
                "cv_minus_test_min": (
                    float(valid_cv_test_gaps.min()) if cv_available else float("nan")
                ),
                "cv_minus_test_max": (
                    float(valid_cv_test_gaps.max()) if cv_available else float("nan")
                ),
                "abs_cv_minus_test_mean": (
                    float(valid_cv_test_gaps.abs().mean())
                    if cv_available
                    else float("nan")
                ),
                "abs_cv_minus_test_max": (
                    float(valid_cv_test_gaps.abs().max())
                    if cv_available
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        ["model_name", "metric_name"],
        ignore_index=True,
    )


__all__ = [
    "CANONICAL_STAGE_ORDER",
    "build_gap_diagnostics_frame",
    "build_generalization_stage_frame",
    "build_generalization_summary_frame",
]
