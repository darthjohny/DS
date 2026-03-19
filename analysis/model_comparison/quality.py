"""Threshold-based quality comparison-layer для задачи `host vs field`.

Модуль добавляет классический quality-блок поверх scored frame-ов:

- выбирает classification threshold только на `train` split;
- считает confusion matrix;
- считает `precision`, `recall`, `f1`, `specificity`,
  `balanced_accuracy` и `accuracy`;
- собирает overall и class-wise quality-таблицы.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from analysis.model_comparison.contracts import (
    DEFAULT_BENCHMARK_SOURCES,
    DEFAULT_QUALITY_CONFIG,
    BenchmarkSources,
    ConfusionMatrixSummary,
    ModelScoreFrames,
    ModelThresholdSummary,
    QualityConfig,
    QualityMetricsSummary,
    QualityRefitMetric,
    QualityScope,
    QualitySplitName,
)
from analysis.model_comparison.metrics import validate_scored_frame


def safe_binary_ratio(numerator: int, denominator: int) -> float:
    """Посчитать бинарную долю с безопасным fallback к `0.0`."""
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def build_threshold_candidates(
    scored_df: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> list[float]:
    """Собрать детерминированный список threshold-кандидатов.

    Для первой волны достаточно unique score values в порядке убывания:
    прогноз меняется только в этих точках.
    """
    validated = validate_scored_frame(scored_df, sources=sources)
    if validated.empty:
        raise ValueError("Threshold selection requires a non-empty scored frame.")

    return sorted(
        validated["model_score"].astype(float).drop_duplicates().tolist(),
        reverse=True,
    )


def iter_threshold_scan(
    scored_df: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> list[tuple[float, int, int, int, int]]:
    """Собрать confusion counts на всех уникальных threshold без O(n^2).

    Идея:

    - сортируем scored frame по score по убыванию;
    - считаем cumulative TP/FP;
    - оцениваем threshold только на последних индексах каждого unique score.
    """
    validated = validate_scored_frame(scored_df, sources=sources)
    if validated.empty:
        raise ValueError("Threshold scan requires a non-empty scored frame.")

    ordered = validated.sort_values(
        "model_score",
        ascending=False,
        kind="mergesort",
        ignore_index=True,
    )
    y_true = ordered[sources.population_col].astype(bool)
    scores = ordered["model_score"].astype(float)

    cumulative_tp = y_true.astype(int).cumsum()
    cumulative_fp = (~y_true).astype(int).cumsum()
    n_host_total = int(y_true.sum())
    n_field_total = int((~y_true).sum())

    result: list[tuple[float, int, int, int, int]] = []
    next_scores = scores.shift(-1)
    is_last_for_score = scores.ne(next_scores)
    last_indices = [
        index
        for index, is_last in enumerate(is_last_for_score.tolist())
        if bool(is_last)
    ]
    for last_index in last_indices:
        threshold = float(scores.iloc[last_index])
        tp = int(cumulative_tp.iloc[last_index])
        fp = int(cumulative_fp.iloc[last_index])
        tn = n_field_total - fp
        fn = n_host_total - tp
        result.append((threshold, tp, fp, tn, fn))
    return result


def compute_confusion_counts(
    scored_df: pd.DataFrame,
    *,
    threshold: float,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> tuple[int, int, int, int]:
    """Посчитать `TP/FP/TN/FN` для scored frame и заданного threshold."""
    validated = validate_scored_frame(scored_df, sources=sources)
    y_true = validated[sources.population_col].astype(bool)
    y_pred = validated["model_score"].astype(float) >= float(threshold)

    tp = int((y_true & y_pred).sum())
    fp = int(((~y_true) & y_pred).sum())
    tn = int(((~y_true) & (~y_pred)).sum())
    fn = int((y_true & (~y_pred)).sum())
    return tp, fp, tn, fn


def compute_quality_metrics(
    *,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> dict[str, float]:
    """Посчитать quality-метрики из confusion counts."""
    precision = safe_binary_ratio(tp, tp + fp)
    recall = safe_binary_ratio(tp, tp + fn)
    specificity = safe_binary_ratio(tn, tn + fp)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    balanced_accuracy = (recall + specificity) / 2.0
    accuracy = safe_binary_ratio(tp + tn, tp + fp + tn + fn)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy),
    }


def select_quality_metric(
    metrics: dict[str, float],
    *,
    metric_name: QualityRefitMetric,
) -> float:
    """Извлечь целевую quality-метрику для threshold selection."""
    if metric_name == "f1":
        return float(metrics["f1"])
    raise ValueError(f"Unsupported quality refit metric: {metric_name}")


def build_quality_record(
    scored_df: pd.DataFrame,
    *,
    split_name: QualitySplitName,
    threshold_summary: ModelThresholdSummary,
    quality_scope: QualityScope = "overall",
    spec_class: str | None = None,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> QualityMetricsSummary:
    """Собрать threshold-based quality для одного scored frame."""
    validated = validate_scored_frame(scored_df, sources=sources)
    if validated.empty:
        raise ValueError("Cannot build quality metrics for an empty scored frame.")

    model_names = validated["model_name"].dropna().unique().tolist()
    if len(model_names) != 1:
        raise ValueError(
            "Each scored frame must contain exactly one model_name. "
            f"Got: {model_names}"
        )

    model_name = str(model_names[0])
    if model_name != threshold_summary.model_name:
        raise ValueError(
            "Threshold summary model_name must match scored frame model_name. "
            f"Got {threshold_summary.model_name!r} and {model_name!r}."
        )

    tp, fp, tn, fn = compute_confusion_counts(
        validated,
        threshold=threshold_summary.threshold_value,
        sources=sources,
    )
    metrics = compute_quality_metrics(tp=tp, fp=fp, tn=tn, fn=fn)

    return QualityMetricsSummary(
        model_name=model_name,
        split_name=split_name,
        quality_scope=quality_scope,
        spec_class=spec_class,
        threshold_metric=threshold_summary.threshold_metric,
        threshold_value=threshold_summary.threshold_value,
        n_rows=int(validated.shape[0]),
        n_host=int(validated[sources.population_col].astype(bool).sum()),
        n_field=int((~validated[sources.population_col].astype(bool)).sum()),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        specificity=metrics["specificity"],
        balanced_accuracy=metrics["balanced_accuracy"],
        accuracy=metrics["accuracy"],
    )


def select_model_threshold(
    scored_df: pd.DataFrame,
    *,
    quality_config: QualityConfig = DEFAULT_QUALITY_CONFIG,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> ModelThresholdSummary:
    """Выбрать classification threshold только на `train` split."""
    validated = validate_scored_frame(scored_df, sources=sources)
    if validated.empty:
        raise ValueError("Threshold selection requires a non-empty scored frame.")

    model_names = validated["model_name"].dropna().unique().tolist()
    if len(model_names) != 1:
        raise ValueError(
            "Each scored frame must contain exactly one model_name. "
            f"Got: {model_names}"
        )

    best_summary: ModelThresholdSummary | None = None
    for threshold, tp, fp, tn, fn in iter_threshold_scan(
        validated,
        sources=sources,
    ):
        metrics = compute_quality_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
        threshold_score = select_quality_metric(
            metrics,
            metric_name=quality_config.refit_metric,
        )
        candidate = ModelThresholdSummary(
            model_name=str(model_names[0]),
            threshold_metric=quality_config.refit_metric,
            threshold_source_split="train",
            threshold_value=float(threshold),
            threshold_score=float(threshold_score),
            n_rows=int(validated.shape[0]),
            n_host=int(validated[sources.population_col].astype(bool).sum()),
            n_field=int((~validated[sources.population_col].astype(bool)).sum()),
        )
        if best_summary is None or candidate.threshold_score > best_summary.threshold_score:
            best_summary = candidate

    if best_summary is None:
        raise ValueError("Threshold selection failed to produce any candidate.")
    return best_summary


def build_classwise_quality_records(
    scored_df: pd.DataFrame,
    *,
    split_name: QualitySplitName,
    threshold_summary: ModelThresholdSummary,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> list[QualityMetricsSummary]:
    """Собрать class-wise quality-records для одного scored frame."""
    validated = validate_scored_frame(scored_df, sources=sources)
    records: list[QualityMetricsSummary] = []

    for spec_class in sources.allowed_classes:
        class_df = validated[validated[sources.class_col] == spec_class].copy()
        if class_df.empty:
            continue
        records.append(
            build_quality_record(
                class_df,
                split_name=split_name,
                threshold_summary=threshold_summary,
                quality_scope="classwise",
                spec_class=spec_class,
                sources=sources,
            )
        )
    return records


def build_confusion_matrix_records(
    scored_df: pd.DataFrame,
    *,
    split_name: QualitySplitName,
    threshold_summary: ModelThresholdSummary,
    quality_scope: QualityScope = "overall",
    spec_class: str | None = None,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> list[ConfusionMatrixSummary]:
    """Собрать long-form confusion-matrix rows для одного scored frame."""
    validated = validate_scored_frame(scored_df, sources=sources)
    model_name = str(validated["model_name"].dropna().iloc[0])
    tp, fp, tn, fn = compute_confusion_counts(
        validated,
        threshold=threshold_summary.threshold_value,
        sources=sources,
    )
    return [
        ConfusionMatrixSummary(
            model_name=model_name,
            split_name=split_name,
            quality_scope=quality_scope,
            spec_class=spec_class,
            threshold_metric=threshold_summary.threshold_metric,
            threshold_value=threshold_summary.threshold_value,
            actual_label=True,
            predicted_label=True,
            n_rows=tp,
        ),
        ConfusionMatrixSummary(
            model_name=model_name,
            split_name=split_name,
            quality_scope=quality_scope,
            spec_class=spec_class,
            threshold_metric=threshold_summary.threshold_metric,
            threshold_value=threshold_summary.threshold_value,
            actual_label=False,
            predicted_label=True,
            n_rows=fp,
        ),
        ConfusionMatrixSummary(
            model_name=model_name,
            split_name=split_name,
            quality_scope=quality_scope,
            spec_class=spec_class,
            threshold_metric=threshold_summary.threshold_metric,
            threshold_value=threshold_summary.threshold_value,
            actual_label=False,
            predicted_label=False,
            n_rows=tn,
        ),
        ConfusionMatrixSummary(
            model_name=model_name,
            split_name=split_name,
            quality_scope=quality_scope,
            spec_class=spec_class,
            threshold_metric=threshold_summary.threshold_metric,
            threshold_value=threshold_summary.threshold_value,
            actual_label=True,
            predicted_label=False,
            n_rows=fn,
        ),
    ]


def build_threshold_summary_frame(
    threshold_summaries: list[ModelThresholdSummary],
) -> pd.DataFrame:
    """Построить tabular summary для набора model-threshold records."""
    return pd.DataFrame.from_records(asdict(summary) for summary in threshold_summaries)


def build_quality_summary_frame(
    scored_split: ModelScoreFrames,
    *,
    threshold_summary: ModelThresholdSummary | None = None,
    quality_config: QualityConfig = DEFAULT_QUALITY_CONFIG,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Построить overall quality summary для train/test частей модели."""
    selected_threshold = (
        select_model_threshold(
            scored_split.train_scored_df,
            quality_config=quality_config,
            sources=sources,
        )
        if threshold_summary is None
        else threshold_summary
    )
    records = [
        build_quality_record(
            scored_split.train_scored_df,
            split_name="train",
            threshold_summary=selected_threshold,
            sources=sources,
        ),
        build_quality_record(
            scored_split.test_scored_df,
            split_name="test",
            threshold_summary=selected_threshold,
            sources=sources,
        ),
    ]
    return pd.DataFrame.from_records(asdict(record) for record in records)


def build_quality_classwise_frame(
    scored_split: ModelScoreFrames,
    *,
    threshold_summary: ModelThresholdSummary | None = None,
    quality_config: QualityConfig = DEFAULT_QUALITY_CONFIG,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Построить class-wise quality summary для train/test частей модели."""
    selected_threshold = (
        select_model_threshold(
            scored_split.train_scored_df,
            quality_config=quality_config,
            sources=sources,
        )
        if threshold_summary is None
        else threshold_summary
    )
    records = [
        *build_classwise_quality_records(
            scored_split.train_scored_df,
            split_name="train",
            threshold_summary=selected_threshold,
            sources=sources,
        ),
        *build_classwise_quality_records(
            scored_split.test_scored_df,
            split_name="test",
            threshold_summary=selected_threshold,
            sources=sources,
        ),
    ]
    return pd.DataFrame.from_records(asdict(record) for record in records)


def build_confusion_matrix_frame(
    scored_split: ModelScoreFrames,
    *,
    threshold_summary: ModelThresholdSummary | None = None,
    quality_config: QualityConfig = DEFAULT_QUALITY_CONFIG,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Построить long-form confusion matrix для train/test частей модели."""
    selected_threshold = (
        select_model_threshold(
            scored_split.train_scored_df,
            quality_config=quality_config,
            sources=sources,
        )
        if threshold_summary is None
        else threshold_summary
    )
    records = [
        *build_confusion_matrix_records(
            scored_split.train_scored_df,
            split_name="train",
            threshold_summary=selected_threshold,
            sources=sources,
        ),
        *build_confusion_matrix_records(
            scored_split.test_scored_df,
            split_name="test",
            threshold_summary=selected_threshold,
            sources=sources,
        ),
    ]
    return pd.DataFrame.from_records(asdict(record) for record in records)


__all__ = [
    "build_classwise_quality_records",
    "build_confusion_matrix_frame",
    "build_confusion_matrix_records",
    "build_quality_classwise_frame",
    "build_quality_record",
    "build_quality_summary_frame",
    "build_threshold_candidates",
    "iter_threshold_scan",
    "build_threshold_summary_frame",
    "compute_confusion_counts",
    "compute_quality_metrics",
    "safe_binary_ratio",
    "select_model_threshold",
    "select_quality_metric",
]
