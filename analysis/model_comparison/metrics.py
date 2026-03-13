"""Метрики comparison-layer для задачи `host vs field`.

Модуль считает единый набор supervised-метрик для scored frame-ов,
которые возвращают baseline-wrapper-ы и будущая основная contrastive
модель.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from analysis.model_comparison.contracts import (
    DEFAULT_BENCHMARK_SOURCES,
    BenchmarkSources,
    ModelScoreFrames,
)


@dataclass(frozen=True, slots=True)
class ModelMetricsRecord:
    """Агрегированная сводка метрик одной модели на одном split."""

    model_name: str
    split_name: str
    n_rows: int
    n_host: int
    n_field: int
    effective_k: int
    roc_auc: float
    pr_auc: float
    brier: float
    precision_at_k: float


@dataclass(frozen=True, slots=True)
class ClasswiseMetricsRecord:
    """Class-wise метрики одной модели на одном split."""

    model_name: str
    split_name: str
    spec_class: str
    n_rows: int
    n_host: int
    n_field: int
    effective_k: int
    roc_auc: float
    pr_auc: float
    brier: float
    precision_at_k: float


def validate_scored_frame(
    scored_df: pd.DataFrame,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Проверить scored frame перед расчётом метрик."""
    required_columns = [
        "model_name",
        "model_score",
        sources.source_id_col,
        sources.class_col,
        sources.population_col,
    ]
    missing_columns = [
        column for column in required_columns if column not in scored_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Scored frame is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    result = scored_df.copy()
    result[sources.class_col] = (
        result[sources.class_col].astype(str).str.strip().str.upper()
    )
    result[sources.population_col] = result[sources.population_col].astype(bool)
    result["model_name"] = result["model_name"].astype(str)
    result["model_score"] = result["model_score"].astype(float)
    return result


def safe_roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Посчитать ROC-AUC или вернуть `nan` для вырожденного случая."""
    if y_true.nunique(dropna=False) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Посчитать PR-AUC или вернуть `nan` для вырожденного случая."""
    if y_true.nunique(dropna=False) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_brier_score(y_true: pd.Series, y_score: pd.Series) -> float:
    """Посчитать Brier score для bounded score в диапазоне [0, 1]."""
    if y_true.empty:
        return float("nan")
    return float(brier_score_loss(y_true, y_score))


def precision_at_k(
    scored_df: pd.DataFrame,
    *,
    k: int,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> tuple[float, int]:
    """Посчитать precision@k для scored frame."""
    validated = validate_scored_frame(scored_df, sources=sources)
    if k <= 0:
        raise ValueError("precision_at_k expects k > 0.")
    if validated.empty:
        return float("nan"), 0

    effective_k = min(int(k), int(validated.shape[0]))
    top_k = validated.sort_values(
        "model_score",
        ascending=False,
        kind="mergesort",
        ignore_index=True,
    ).head(effective_k)
    precision_value = float(top_k[sources.population_col].astype(float).mean())
    return precision_value, effective_k


def build_metrics_record(
    scored_df: pd.DataFrame,
    *,
    split_name: str,
    precision_k: int,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> ModelMetricsRecord:
    """Собрать сводку метрик одной модели на одном split."""
    validated = validate_scored_frame(scored_df, sources=sources)
    if validated.empty:
        raise ValueError("Cannot build metrics for an empty scored frame.")

    model_names = validated["model_name"].dropna().unique().tolist()
    if len(model_names) != 1:
        raise ValueError(
            "Each scored frame must contain exactly one model_name. "
            f"Got: {model_names}"
        )

    y_true = validated[sources.population_col].astype(int)
    y_score = validated["model_score"]
    precision_value, effective_k = precision_at_k(
        validated,
        k=precision_k,
        sources=sources,
    )
    return ModelMetricsRecord(
        model_name=str(model_names[0]),
        split_name=str(split_name),
        n_rows=int(validated.shape[0]),
        n_host=int(validated[sources.population_col].sum()),
        n_field=int((~validated[sources.population_col]).sum()),
        effective_k=effective_k,
        roc_auc=safe_roc_auc(y_true, y_score),
        pr_auc=safe_pr_auc(y_true, y_score),
        brier=safe_brier_score(y_true, y_score),
        precision_at_k=precision_value,
    )


def build_classwise_metrics_records(
    scored_df: pd.DataFrame,
    *,
    split_name: str,
    precision_k: int,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> list[ClasswiseMetricsRecord]:
    """Собрать class-wise метрики одной модели на одном split."""
    validated = validate_scored_frame(scored_df, sources=sources)
    records: list[ClasswiseMetricsRecord] = []

    for spec_class in sources.allowed_classes:
        class_df = validated[validated[sources.class_col] == spec_class].copy()
        if class_df.empty:
            continue

        overall_record = build_metrics_record(
            class_df,
            split_name=split_name,
            precision_k=precision_k,
            sources=sources,
        )
        records.append(
            ClasswiseMetricsRecord(
                model_name=overall_record.model_name,
                split_name=overall_record.split_name,
                spec_class=spec_class,
                n_rows=overall_record.n_rows,
                n_host=overall_record.n_host,
                n_field=overall_record.n_field,
                effective_k=overall_record.effective_k,
                roc_auc=overall_record.roc_auc,
                pr_auc=overall_record.pr_auc,
                brier=overall_record.brier,
                precision_at_k=overall_record.precision_at_k,
            )
        )

    return records


def build_metrics_frame(
    scored_split: ModelScoreFrames,
    *,
    precision_k: int = 50,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Построить общую metrics-таблицу для train/test частей модели."""
    records = [
        build_metrics_record(
            scored_split.train_scored_df,
            split_name="train",
            precision_k=precision_k,
            sources=sources,
        ),
        build_metrics_record(
            scored_split.test_scored_df,
            split_name="test",
            precision_k=precision_k,
            sources=sources,
        ),
    ]
    return pd.DataFrame.from_records(asdict(record) for record in records)


def build_classwise_metrics_frame(
    scored_split: ModelScoreFrames,
    *,
    precision_k: int = 50,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Построить class-wise metrics-таблицу для train/test частей модели."""
    records = [
        *build_classwise_metrics_records(
            scored_split.train_scored_df,
            split_name="train",
            precision_k=precision_k,
            sources=sources,
        ),
        *build_classwise_metrics_records(
            scored_split.test_scored_df,
            split_name="test",
            precision_k=precision_k,
            sources=sources,
        ),
    ]
    return pd.DataFrame.from_records(asdict(record) for record in records)


def format_metric_value(value: object) -> str:
    """Преобразовать значение метрики в компактный текст."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


__all__ = [
    "ClasswiseMetricsRecord",
    "ModelMetricsRecord",
    "build_classwise_metrics_frame",
    "build_classwise_metrics_records",
    "build_metrics_frame",
    "build_metrics_record",
    "format_metric_value",
    "precision_at_k",
    "safe_brier_score",
    "safe_pr_auc",
    "safe_roc_auc",
    "validate_scored_frame",
]
