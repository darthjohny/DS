"""Repeated split evaluation для heavy validation слоя.

Модуль запускает comparison-layer на нескольких `random_state` и собирает:

- long-form per-split diagnostics по моделям и метрикам;
- агрегированную summary устойчивости по каждой модели;
- typed result, пригодный для дальнейшего risk audit слоя.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import pandas as pd

from analysis.model_comparison.contracts import (
    ClassSearchSummary,
    ComparisonProtocol,
    ModelSearchSummary,
)
from analysis.model_comparison.contrastive import run_main_contrastive_model
from analysis.model_comparison.data import load_and_split_benchmark_dataset
from analysis.model_comparison.legacy_gaussian import run_legacy_gaussian_baseline
from analysis.model_comparison.mlp_baseline import run_mlp_baseline
from analysis.model_comparison.random_forest import run_random_forest_baseline
from analysis.model_comparison.reporting import (
    build_comparison_summary_frame,
    build_generalization_diagnostics_frame,
    build_search_summary_frame,
)
from analysis.model_comparison.validation import validate_benchmark_split
from analysis.model_validation.contracts import (
    ModelValidationProtocol,
    ModelValidationSplitResult,
    RepeatedSplitEvaluationResult,
)
from analysis.model_validation.generalization_summary import (
    build_gap_diagnostics_frame,
    build_generalization_stage_frame,
    build_generalization_summary_frame,
)
from analysis.model_validation.risk_audit import build_model_risk_audit_frame
from analysis.model_validation.scalars import scalar_to_float, scalar_to_int

type SplitRunner = Callable[[ComparisonProtocol], ModelValidationSplitResult]
type SearchSummaryEntry = ClassSearchSummary | ModelSearchSummary

PRIMARY_METRIC_DIRECTIONS: dict[str, str] = {
    "roc_auc": "maximize",
    "pr_auc": "maximize",
    "brier": "minimize",
    "precision_at_k": "maximize",
}
def build_split_protocol(
    protocol: ModelValidationProtocol,
    *,
    random_state: int,
) -> ComparisonProtocol:
    """Собрать comparison protocol для одного repeated split."""
    return replace(
        protocol.comparison_protocol,
        split=replace(
            protocol.comparison_protocol.split,
            random_state=int(random_state),
        ),
    )


def run_validation_split(
    comparison_protocol: ComparisonProtocol,
) -> ModelValidationSplitResult:
    """Запустить один comparison split без записи benchmark-артефактов."""
    split = load_and_split_benchmark_dataset(protocol=comparison_protocol)
    validation_result = validate_benchmark_split(
        split,
        protocol=comparison_protocol,
    )
    if validation_result.has_errors:
        joined_errors = "; ".join(validation_result.errors)
        raise ValueError(
            "Heavy validation repeated split failed benchmark validation: "
            f"{joined_errors}"
        )

    main_run = run_main_contrastive_model(
        split,
        sources=comparison_protocol.sources,
        cv_config=comparison_protocol.cv,
        search_config=comparison_protocol.search,
    )
    legacy_run = run_legacy_gaussian_baseline(
        split,
        sources=comparison_protocol.sources,
        cv_config=comparison_protocol.cv,
        search_config=comparison_protocol.search,
    )
    mlp_run = run_mlp_baseline(
        split,
        sources=comparison_protocol.sources,
        cv_config=comparison_protocol.cv,
        search_config=comparison_protocol.search,
    )
    random_forest_run = run_random_forest_baseline(
        split,
        sources=comparison_protocol.sources,
        cv_config=comparison_protocol.cv,
        search_config=comparison_protocol.search,
    )

    scored_splits = [
        main_run.scored_split,
        legacy_run.scored_split,
        mlp_run.scored_split,
        random_forest_run.scored_split,
    ]
    search_summaries: list[SearchSummaryEntry] = [
        main_run.search_summary,
        legacy_run.search_summary,
        *mlp_run.search_results_by_class.values(),
        *random_forest_run.search_results_by_class.values(),
    ]
    summary_df = build_comparison_summary_frame(
        scored_splits,
        precision_k=comparison_protocol.search.precision_k,
        protocol=comparison_protocol,
    )
    search_summary_df = build_search_summary_frame(search_summaries)
    generalization_df = build_generalization_diagnostics_frame(
        summary_df,
        search_summary_df=search_summary_df,
    )
    return ModelValidationSplitResult(
        split_random_state=comparison_protocol.split.random_state,
        summary_df=summary_df,
        search_summary_df=search_summary_df,
        generalization_df=generalization_df,
    )


def build_repeated_splits_frame(
    split_results: Sequence[ModelValidationSplitResult],
    *,
    validation_protocol: ModelValidationProtocol,
) -> pd.DataFrame:
    """Собрать long-form per-split таблицу для heavy validation."""
    frames: list[pd.DataFrame] = []
    for split_result in split_results:
        if split_result.generalization_df.empty:
            continue
        split_frame = split_result.generalization_df.copy()
        split_frame.insert(0, "split_random_state", int(split_result.split_random_state))
        split_frame.insert(1, "validation_protocol_name", validation_protocol.name)
        split_frame.insert(
            2,
            "benchmark_protocol_name",
            validation_protocol.comparison_protocol.name,
        )
        split_frame.insert(
            3,
            "precision_k",
            int(validation_protocol.comparison_protocol.search.precision_k),
        )
        split_frame.insert(
            4,
            "metric_direction",
            split_frame["metric_name"].astype(str).map(PRIMARY_METRIC_DIRECTIONS),
        )
        frames.append(split_frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["model_name", "metric_name", "split_random_state"],
        ignore_index=True,
    )


def build_repeated_split_model_summary(
    repeated_splits_df: pd.DataFrame,
) -> pd.DataFrame:
    """Агрегировать repeated split diagnostics по моделям и метрикам."""
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
        metric_direction = str(group["metric_direction"].iloc[0])
        test_series = group["test_value"].astype(float)
        train_series = group["train_value"].astype(float)
        gap_series = group["train_minus_test"].astype(float)
        abs_gap_series = group["abs_train_test_gap"].astype(float)
        cv_gap_series = group["cv_minus_test"].astype(float)
        cv_mean_series = group["cv_score_mean"].astype(float)
        valid_cv_mask = ~cv_gap_series.isna()
        valid_cv_gaps = cv_gap_series[valid_cv_mask]
        valid_cv_means = cv_mean_series[valid_cv_mask]

        if metric_direction == "minimize":
            best_idx = test_series.idxmin()
            worst_idx = test_series.idxmax()
        else:
            best_idx = test_series.idxmax()
            worst_idx = test_series.idxmin()

        rows.append(
            {
                "validation_protocol_name": str(validation_protocol_name),
                "benchmark_protocol_name": str(benchmark_protocol_name),
                "model_name": str(model_name),
                "metric_name": str(metric_name),
                "metric_direction": metric_direction,
                "precision_k": scalar_to_int(group["precision_k"].iloc[0]),
                "split_count": int(group.shape[0]),
                "refit_split_count": int(group["is_refit_metric"].astype(bool).sum()),
                "cv_available_splits": int(valid_cv_mask.sum()),
                "train_mean": float(train_series.mean()),
                "train_std": float(train_series.std(ddof=0)),
                "test_mean": float(test_series.mean()),
                "test_std": float(test_series.std(ddof=0)),
                "test_min": float(test_series.min()),
                "test_max": float(test_series.max()),
                "best_split_random_state": scalar_to_int(
                    group.loc[best_idx, "split_random_state"]
                ),
                "best_test_value": scalar_to_float(group.loc[best_idx, "test_value"]),
                "worst_split_random_state": scalar_to_int(
                    group.loc[worst_idx, "split_random_state"]
                ),
                "worst_test_value": scalar_to_float(group.loc[worst_idx, "test_value"]),
                "train_minus_test_mean": float(gap_series.mean()),
                "abs_train_test_gap_mean": float(abs_gap_series.mean()),
                "abs_train_test_gap_max": float(abs_gap_series.max()),
                "cv_score_mean": (
                    float(valid_cv_means.mean())
                    if not valid_cv_means.empty
                    else float("nan")
                ),
                "cv_score_std": (
                    float(valid_cv_means.std(ddof=0))
                    if len(valid_cv_means) > 0
                    else float("nan")
                ),
                "cv_minus_test_mean": (
                    float(valid_cv_gaps.mean())
                    if not valid_cv_gaps.empty
                    else float("nan")
                ),
                "abs_cv_minus_test_mean": (
                    float(valid_cv_gaps.abs().mean())
                    if not valid_cv_gaps.empty
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        ["model_name", "metric_name"],
        ignore_index=True,
    )


def run_repeated_split_evaluation(
    protocol: ModelValidationProtocol,
    *,
    run_split: SplitRunner | None = None,
) -> RepeatedSplitEvaluationResult:
    """Запустить repeated split evaluation и собрать агрегированные артефакты."""
    split_runner = run_validation_split if run_split is None else run_split
    split_results = tuple(
        split_runner(build_split_protocol(protocol, random_state=random_state))
        for random_state in protocol.repeated_split.random_states
    )
    repeated_splits_df = build_repeated_splits_frame(
        split_results,
        validation_protocol=protocol,
    )
    model_summary_df = build_repeated_split_model_summary(repeated_splits_df)
    generalization_stage_df = build_generalization_stage_frame(repeated_splits_df)
    generalization_summary_df = build_generalization_summary_frame(
        generalization_stage_df
    )
    gap_diagnostics_df = build_gap_diagnostics_frame(repeated_splits_df)
    risk_audit_df = build_model_risk_audit_frame(
        protocol,
        generalization_summary_df=generalization_summary_df,
        gap_diagnostics_df=gap_diagnostics_df,
    )
    return RepeatedSplitEvaluationResult(
        split_results=split_results,
        repeated_splits_df=repeated_splits_df,
        model_summary_df=model_summary_df,
        generalization_summary_df=generalization_summary_df,
        gap_diagnostics_df=gap_diagnostics_df,
        risk_audit_df=risk_audit_df,
    )


__all__ = [
    "PRIMARY_METRIC_DIRECTIONS",
    "SplitRunner",
    "build_repeated_split_model_summary",
    "build_repeated_splits_frame",
    "build_split_protocol",
    "run_repeated_split_evaluation",
    "run_validation_split",
    "scalar_to_float",
]
