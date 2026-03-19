"""Preflight validation для benchmark dataset и split comparison-layer.

Модуль нужен для явного dataset gate до model fitting:

- проверяет целостность `full/train/test` split;
- ловит overlap между train и test;
- проверяет сохранение stratify-лейблов;
- собирает простые drift/balance diagnostics;
- пишет воспроизводимые markdown/CSV-артефакты.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path

import pandas as pd

from analysis.model_comparison.contracts import (
    DEFAULT_COMPARISON_PROTOCOL,
    BenchmarkSources,
    ComparisonProtocol,
)
from analysis.model_comparison.data import BenchmarkSplit
from analysis.model_comparison.tuning import (
    build_stratify_labels,
    validate_cross_validation_inputs,
)

DEFAULT_LABEL_SHARE_WARNING_THRESHOLD = 0.15
DEFAULT_FEATURE_DRIFT_WARNING_THRESHOLD = 1.0


@dataclass(slots=True)
class BenchmarkValidationResult:
    """Сводка preflight validation для benchmark split."""

    summary_df: pd.DataFrame
    stratify_df: pd.DataFrame
    feature_drift_df: pd.DataFrame
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def has_errors(self) -> bool:
        """Показать, есть ли блокирующие validation-ошибки."""
        return bool(self.errors)


def frame_to_text(df: pd.DataFrame) -> str:
    """Преобразовать DataFrame в компактный текстовый блок для markdown."""
    if df.empty:
        return "Пусто"
    return df.to_string(index=False)


def scalar_to_float(value: object) -> float:
    """Преобразовать pandas-скаляр в `float` с явной runtime-проверкой."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"Value is not float-compatible: {value!r}")


def build_stratify_balance_frame(
    split: BenchmarkSplit,
    *,
    sources: BenchmarkSources,
) -> pd.DataFrame:
    """Собрать распределение stratify-групп по full/train/test."""

    def scope_frame(df_part: pd.DataFrame, *, scope_name: str) -> pd.DataFrame:
        labels = build_stratify_labels(df_part, sources=sources)
        rows = labels.value_counts().rename_axis("stratify_label").reset_index(name="n_rows")
        rows["scope_name"] = scope_name
        rows["share"] = rows["n_rows"].astype(float) / float(max(len(df_part), 1))
        rows[sources.class_col] = rows["stratify_label"].str.split("|").str[0]
        rows[sources.population_col] = (
            rows["stratify_label"].str.split("|").str[1].astype(int).astype(bool)
        )
        return rows.loc[
            :,
            [
                "scope_name",
                "stratify_label",
                sources.class_col,
                sources.population_col,
                "n_rows",
                "share",
            ],
        ]

    combined = pd.concat(
        [
            scope_frame(split.full_df, scope_name="full"),
            scope_frame(split.train_df, scope_name="train"),
            scope_frame(split.test_df, scope_name="test"),
        ],
        ignore_index=True,
        sort=False,
    )
    return combined.sort_values(
        ["stratify_label", "scope_name"],
        ignore_index=True,
    )


def build_feature_drift_frame(
    split: BenchmarkSplit,
    *,
    sources: BenchmarkSources,
) -> pd.DataFrame:
    """Собрать простую drift-диагностику по train/test feature means."""
    rows: list[dict[str, float | str]] = []
    for feature_name in sources.feature_columns:
        train_series = split.train_df[feature_name].astype(float)
        test_series = split.test_df[feature_name].astype(float)
        train_mean = float(train_series.mean())
        test_mean = float(test_series.mean())
        train_std = float(train_series.std(ddof=0))
        test_std = float(test_series.std(ddof=0))
        pooled_std = ((train_std**2 + test_std**2) / 2.0) ** 0.5
        if pooled_std == 0.0:
            standardized_mean_diff = 0.0 if train_mean == test_mean else float("inf")
        else:
            standardized_mean_diff = abs(train_mean - test_mean) / pooled_std

        rows.append(
            {
                "feature_name": feature_name,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "test_std": test_std,
                "pooled_std": float(pooled_std),
                "abs_standardized_mean_diff": float(standardized_mean_diff),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        "feature_name",
        ignore_index=True,
    )


def build_validation_summary_frame(
    split: BenchmarkSplit,
    *,
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    stratify_df: pd.DataFrame,
    feature_drift_df: pd.DataFrame,
    sources: BenchmarkSources,
) -> pd.DataFrame:
    """Собрать агрегированную summary-таблицу dataset gate."""
    train_host_share = float(split.train_df[sources.population_col].astype(bool).mean())
    test_host_share = float(split.test_df[sources.population_col].astype(bool).mean())
    full_host_share = float(split.full_df[sources.population_col].astype(bool).mean())

    pivot = stratify_df.pivot(
        index="stratify_label",
        columns="scope_name",
        values="share",
    ).fillna(0.0)
    max_label_share_gap = float((pivot["train"] - pivot["test"]).abs().max())
    max_feature_drift = float(feature_drift_df["abs_standardized_mean_diff"].max())

    return pd.DataFrame(
        [
            {
                "full_rows": int(split.full_df.shape[0]),
                "train_rows": int(split.train_df.shape[0]),
                "test_rows": int(split.test_df.shape[0]),
                "n_classes": int(split.full_df[sources.class_col].nunique()),
                "n_stratify_labels": int(stratify_df["stratify_label"].nunique()),
                "full_host_share": full_host_share,
                "train_host_share": train_host_share,
                "test_host_share": test_host_share,
                "max_abs_label_share_gap": max_label_share_gap,
                "max_abs_feature_smd": max_feature_drift,
                "error_count": len(errors),
                "warning_count": len(warnings),
            }
        ]
    )


def validate_benchmark_split(
    split: BenchmarkSplit,
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
    label_share_warning_threshold: float = DEFAULT_LABEL_SHARE_WARNING_THRESHOLD,
    feature_drift_warning_threshold: float = DEFAULT_FEATURE_DRIFT_WARNING_THRESHOLD,
) -> BenchmarkValidationResult:
    """Проверить benchmark split и собрать preflight diagnostics."""
    sources = protocol.sources
    errors: list[str] = []
    warnings: list[str] = []

    if split.full_df.empty:
        errors.append("Benchmark validation requires a non-empty full split.")
    if split.train_df.empty:
        errors.append("Benchmark validation requires a non-empty train split.")
    if split.test_df.empty:
        errors.append("Benchmark validation requires a non-empty test split.")

    train_ids = set(split.train_df[sources.source_id_col].astype(str))
    test_ids = set(split.test_df[sources.source_id_col].astype(str))
    full_ids = set(split.full_df[sources.source_id_col].astype(str))

    overlap_ids = sorted(train_ids & test_ids)
    if overlap_ids:
        sample = ", ".join(overlap_ids[:5])
        errors.append(
            "Benchmark validation found train/test overlap by source_id. "
            f"Sample ids: {sample}"
        )

    if train_ids | test_ids != full_ids:
        errors.append(
            "Benchmark validation expects train/test union to match full split source_ids."
        )

    full_labels = set(build_stratify_labels(split.full_df, sources=sources).astype(str))
    train_labels = set(build_stratify_labels(split.train_df, sources=sources).astype(str))
    test_labels = set(build_stratify_labels(split.test_df, sources=sources).astype(str))
    if train_labels != full_labels:
        errors.append("Benchmark validation found missing stratify labels in train split.")
    if test_labels != full_labels:
        errors.append("Benchmark validation found missing stratify labels in test split.")

    try:
        validate_cross_validation_inputs(
            split.train_df,
            cv_config=protocol.cv,
            sources=sources,
        )
    except ValueError as exc:
        errors.append(str(exc))

    stratify_df = build_stratify_balance_frame(split, sources=sources)
    feature_drift_df = build_feature_drift_frame(split, sources=sources)
    summary_df = build_validation_summary_frame(
        split,
        errors=tuple(errors),
        warnings=tuple(warnings),
        stratify_df=stratify_df,
        feature_drift_df=feature_drift_df,
        sources=sources,
    )

    max_label_share_gap = scalar_to_float(
        summary_df.at[0, "max_abs_label_share_gap"]
    )
    if max_label_share_gap > label_share_warning_threshold:
        warnings.append(
            "Train/test stratify balance differs more than expected. "
            f"max_abs_label_share_gap={max_label_share_gap:.3f}"
        )

    drifting_features = feature_drift_df[
        feature_drift_df["abs_standardized_mean_diff"] > feature_drift_warning_threshold
    ]
    if not drifting_features.empty:
        sample = ", ".join(drifting_features["feature_name"].astype(str).tolist())
        warnings.append(
            "Train/test feature drift exceeded the warning threshold for: "
            f"{sample}"
        )

    summary_df = build_validation_summary_frame(
        split,
        errors=tuple(errors),
        warnings=tuple(warnings),
        stratify_df=stratify_df,
        feature_drift_df=feature_drift_df,
        sources=sources,
    )
    return BenchmarkValidationResult(
        summary_df=summary_df,
        stratify_df=stratify_df,
        feature_drift_df=feature_drift_df,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def build_validation_markdown(
    result: BenchmarkValidationResult,
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
    note: str = "",
) -> str:
    """Собрать markdown-отчёт для dataset gate."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    note_text = note.strip() or "-"
    error_lines = "\n".join(f"- {message}" for message in result.errors) or "- Пусто"
    warning_lines = "\n".join(f"- {message}" for message in result.warnings) or "- Пусто"
    return f"""# Benchmark Dataset Validation

Дата: {created_at}
Protocol: `{protocol.name}`

## Источники benchmark
- host view: `{protocol.sources.host_view}`
- field view: `{protocol.sources.field_view}`
- test_size: `{protocol.split.test_size:.2f}`
- cv_folds: `{protocol.cv.n_splits}`

## Summary
{frame_to_text(result.summary_df)}

## Errors
{error_lines}

## Warnings
{warning_lines}

## Stratify Balance
{frame_to_text(result.stratify_df)}

## Feature Drift
{frame_to_text(result.feature_drift_df)}

## Примечание
{note_text}
"""


def save_benchmark_validation_artifacts(
    run_name: str,
    result: BenchmarkValidationResult,
    *,
    output_dir: Path,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
    note: str = "",
) -> Path:
    """Сохранить markdown/CSV артефакты dataset gate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / run_name
    markdown_path = prefix.with_name(f"{prefix.name}_dataset_validation.md")
    summary_path = prefix.with_name(f"{prefix.name}_dataset_validation_summary.csv")
    stratify_path = prefix.with_name(f"{prefix.name}_dataset_validation_stratify.csv")
    feature_path = prefix.with_name(f"{prefix.name}_dataset_validation_feature_drift.csv")

    markdown_path.write_text(
        build_validation_markdown(
            result,
            protocol=protocol,
            note=note,
        ),
        encoding="utf-8",
    )
    result.summary_df.to_csv(summary_path, index=False)
    result.stratify_df.to_csv(stratify_path, index=False)
    result.feature_drift_df.to_csv(feature_path, index=False)
    return markdown_path


__all__ = [
    "BenchmarkValidationResult",
    "DEFAULT_FEATURE_DRIFT_WARNING_THRESHOLD",
    "DEFAULT_LABEL_SHARE_WARNING_THRESHOLD",
    "build_feature_drift_frame",
    "build_stratify_balance_frame",
    "build_validation_markdown",
    "build_validation_summary_frame",
    "save_benchmark_validation_artifacts",
    "validate_benchmark_split",
]
