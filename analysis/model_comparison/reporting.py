"""Сборка markdown- и CSV-артефактов comparison-layer."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from analysis.model_comparison.contracts import (
    DEFAULT_COMPARISON_PROTOCOL,
    ClassSearchSummary,
    ComparisonProtocol,
    ModelScoreFrames,
    ModelSearchSummary,
)
from analysis.model_comparison.generalization_audit import (
    build_generalization_audit_markdown,
    build_model_generalization_audit_frame,
)
from analysis.model_comparison.metrics import (
    build_classwise_metrics_frame,
    build_metrics_frame,
)
from analysis.model_comparison.quality import (
    build_confusion_matrix_frame,
    build_quality_classwise_frame,
    build_quality_summary_frame,
    build_threshold_summary_frame,
    select_model_threshold,
)

DEFAULT_MODEL_COMPARISON_OUTPUT_DIR = Path("experiments/model_comparison")

type SearchSummaryEntry = ClassSearchSummary | ModelSearchSummary


def frame_to_text(df: pd.DataFrame) -> str:
    """Преобразовать DataFrame в компактный текстовый блок для markdown."""
    if df.empty:
        return "Пусто"
    return df.to_string(index=False)


def build_comparison_summary_frame(
    scored_splits: list[ModelScoreFrames],
    *,
    precision_k: int = 50,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Собрать общую metrics-таблицу по нескольким моделям."""
    frames = [
        build_metrics_frame(
            scored_split,
            precision_k=precision_k,
            sources=protocol.sources,
        )
        for scored_split in scored_splits
    ]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["split_name", "model_name"], ignore_index=True)
    )


def build_comparison_classwise_frame(
    scored_splits: list[ModelScoreFrames],
    *,
    precision_k: int = 50,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Собрать class-wise metrics-таблицу по нескольким моделям."""
    frames = [
        build_classwise_metrics_frame(
            scored_split,
            precision_k=precision_k,
            sources=protocol.sources,
        )
        for scored_split in scored_splits
    ]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["split_name", "model_name", "spec_class"], ignore_index=True)
    )


def build_comparison_thresholds_frame(
    scored_splits: list[ModelScoreFrames],
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Собрать train-selected threshold summary по нескольким моделям."""
    summaries = [
        select_model_threshold(
            scored_split.train_scored_df,
            quality_config=protocol.quality,
            sources=protocol.sources,
        )
        for scored_split in scored_splits
    ]
    if not summaries:
        return pd.DataFrame()
    return build_threshold_summary_frame(summaries).sort_values(
        "model_name",
        ignore_index=True,
    )


def build_comparison_quality_summary_frame(
    scored_splits: list[ModelScoreFrames],
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Собрать overall quality-таблицу по нескольким моделям."""
    frames = []
    for scored_split in scored_splits:
        threshold_summary = select_model_threshold(
            scored_split.train_scored_df,
            quality_config=protocol.quality,
            sources=protocol.sources,
        )
        frames.append(
            build_quality_summary_frame(
                scored_split,
                threshold_summary=threshold_summary,
                quality_config=protocol.quality,
                sources=protocol.sources,
            )
        )
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["split_name", "model_name"], ignore_index=True)
    )


def build_comparison_quality_classwise_frame(
    scored_splits: list[ModelScoreFrames],
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Собрать class-wise quality-таблицу по нескольким моделям."""
    frames = []
    for scored_split in scored_splits:
        threshold_summary = select_model_threshold(
            scored_split.train_scored_df,
            quality_config=protocol.quality,
            sources=protocol.sources,
        )
        frames.append(
            build_quality_classwise_frame(
                scored_split,
                threshold_summary=threshold_summary,
                quality_config=protocol.quality,
                sources=protocol.sources,
            )
        )
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["split_name", "model_name", "spec_class"], ignore_index=True)
    )


def build_comparison_confusion_matrix_frame(
    scored_splits: list[ModelScoreFrames],
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Собрать long-form confusion matrix по нескольким моделям."""
    frames = []
    for scored_split in scored_splits:
        threshold_summary = select_model_threshold(
            scored_split.train_scored_df,
            quality_config=protocol.quality,
            sources=protocol.sources,
        )
        frames.append(
            build_confusion_matrix_frame(
                scored_split,
                threshold_summary=threshold_summary,
                quality_config=protocol.quality,
                sources=protocol.sources,
            )
        )
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(
            [
                "split_name",
                "model_name",
                "quality_scope",
                "spec_class",
                "actual_label",
                "predicted_label",
            ],
            ignore_index=True,
        )
    )


def build_search_summary_frame(
    search_summaries: Sequence[SearchSummaryEntry],
) -> pd.DataFrame:
    """Собрать табличную сводку hyperparameter search по всем моделям."""
    rows: list[dict[str, object]] = []
    for summary in search_summaries:
        spec_class = summary.spec_class if isinstance(summary, ClassSearchSummary) else None
        rows.append(
            {
                "model_name": summary.model_name,
                "search_scope": "class" if spec_class is not None else "model",
                "spec_class": spec_class,
                "refit_metric": summary.refit_metric,
                "precision_k": summary.precision_k,
                "cv_folds": summary.cv_folds,
                "n_train_rows": summary.n_train_rows,
                "n_host": summary.n_host,
                "n_field": summary.n_field,
                "candidate_count": summary.candidate_count,
                "best_cv_score": summary.best_cv_score,
                "cv_score_std": summary.cv_score_std,
                "cv_score_min": summary.cv_score_min,
                "cv_score_max": summary.cv_score_max,
                "best_params_json": json.dumps(
                    summary.best_params,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame.from_records(rows).sort_values(
        ["model_name", "search_scope", "spec_class"],
        ignore_index=True,
    )


def aggregate_search_summary_by_model(search_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Собрать model-level summary из class/model search records."""
    if search_summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for model_name, group in search_summary_df.groupby("model_name", sort=True):
        weights = group["n_train_rows"].astype(float)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            raise ValueError("Search summary weights must be positive.")

        if group["search_scope"].eq("model").all() and len(group) == 1:
            row = group.iloc[0]
            rows.append(
                {
                    "model_name": str(model_name),
                    "refit_metric": str(row["refit_metric"]),
                    "precision_k": int(row["precision_k"]),
                    "cv_folds": int(row["cv_folds"]),
                    "cv_summary_scope": "model",
                    "cv_score_mean": float(row["best_cv_score"]),
                    "cv_score_std": float(row["cv_score_std"]),
                    "cv_score_min": float(row["cv_score_min"]),
                    "cv_score_max": float(row["cv_score_max"]),
                }
            )
            continue

        rows.append(
            {
                "model_name": str(model_name),
                "refit_metric": str(group["refit_metric"].iloc[0]),
                "precision_k": int(group["precision_k"].iloc[0]),
                "cv_folds": int(group["cv_folds"].iloc[0]),
                "cv_summary_scope": "class_weighted_mean",
                "cv_score_mean": float(
                    (group["best_cv_score"].astype(float) * weights).sum() / weight_sum
                ),
                "cv_score_std": float(
                    (group["cv_score_std"].astype(float) * weights).sum() / weight_sum
                ),
                "cv_score_min": float(group["cv_score_min"].astype(float).min()),
                "cv_score_max": float(group["cv_score_max"].astype(float).max()),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values("model_name", ignore_index=True)


def build_generalization_diagnostics_frame(
    summary_df: pd.DataFrame,
    *,
    search_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Собрать long-form диагностику generalization по моделям и метрикам."""
    if summary_df.empty:
        return pd.DataFrame()

    search_by_model = aggregate_search_summary_by_model(
        pd.DataFrame() if search_summary_df is None else search_summary_df
    )
    search_lookup = (
        search_by_model.set_index("model_name").to_dict(orient="index")
        if not search_by_model.empty
        else {}
    )
    metric_columns = ["roc_auc", "pr_auc", "brier", "precision_at_k"]
    rows: list[dict[str, object]] = []

    for model_name, group in summary_df.groupby("model_name", sort=True):
        split_lookup = {
            str(row["split_name"]): row for _, row in group.iterrows()
        }
        train_row = split_lookup.get("train")
        test_row = split_lookup.get("test")
        if train_row is None or test_row is None:
            raise ValueError(
                "Generalization diagnostics require both train and test rows "
                f"for model {model_name}."
            )

        search_row = search_lookup.get(str(model_name))
        for metric_name in metric_columns:
            train_value = float(train_row[metric_name])
            test_value = float(test_row[metric_name])
            train_test_gap = train_value - test_value
            abs_gap = abs(train_test_gap)
            refit_metric = (
                str(search_row["refit_metric"]) if search_row is not None else ""
            )
            is_refit_metric = metric_name == refit_metric
            cv_mean = (
                float(search_row["cv_score_mean"])
                if search_row is not None and is_refit_metric
                else float("nan")
            )
            cv_std = (
                float(search_row["cv_score_std"])
                if search_row is not None and is_refit_metric
                else float("nan")
            )
            cv_min = (
                float(search_row["cv_score_min"])
                if search_row is not None and is_refit_metric
                else float("nan")
            )
            cv_max = (
                float(search_row["cv_score_max"])
                if search_row is not None and is_refit_metric
                else float("nan")
            )
            cv_test_gap = cv_mean - test_value if not math.isnan(cv_mean) else float("nan")
            rows.append(
                {
                    "model_name": str(model_name),
                    "metric_name": metric_name,
                    "train_scope": "in_sample_refit",
                    "test_scope": "holdout_test",
                    "train_value": train_value,
                    "test_value": test_value,
                    "train_minus_test": train_test_gap,
                    "abs_train_test_gap": abs_gap,
                    "is_refit_metric": bool(is_refit_metric),
                    "cv_summary_scope": (
                        str(search_row["cv_summary_scope"])
                        if search_row is not None and is_refit_metric
                        else None
                    ),
                    "cv_score_mean": cv_mean,
                    "cv_score_std": cv_std,
                    "cv_score_min": cv_min,
                    "cv_score_max": cv_max,
                    "cv_minus_test": cv_test_gap,
                }
            )

    return pd.DataFrame.from_records(rows).sort_values(
        ["model_name", "metric_name"],
        ignore_index=True,
    )


def build_comparison_markdown(
    summary_df: pd.DataFrame,
    classwise_df: pd.DataFrame,
    *,
    thresholds_df: pd.DataFrame | None = None,
    quality_summary_df: pd.DataFrame | None = None,
    quality_classwise_df: pd.DataFrame | None = None,
    confusion_matrix_df: pd.DataFrame | None = None,
    search_summary_df: pd.DataFrame | None = None,
    generalization_df: pd.DataFrame | None = None,
    audit_df: pd.DataFrame | None = None,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
    precision_k: int = 50,
    note: str = "",
) -> str:
    """Собрать markdown summary для comparative benchmark."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    note_text = note.strip() or "-"
    normalized_thresholds_df = pd.DataFrame() if thresholds_df is None else thresholds_df
    normalized_quality_summary_df = (
        pd.DataFrame() if quality_summary_df is None else quality_summary_df
    )
    normalized_quality_classwise_df = (
        pd.DataFrame() if quality_classwise_df is None else quality_classwise_df
    )
    normalized_confusion_matrix_df = (
        pd.DataFrame() if confusion_matrix_df is None else confusion_matrix_df
    )
    normalized_search_summary_df = (
        pd.DataFrame() if search_summary_df is None else search_summary_df
    )
    normalized_generalization_df = (
        pd.DataFrame() if generalization_df is None else generalization_df
    )
    normalized_audit_df = pd.DataFrame() if audit_df is None else audit_df
    model_lines = "\n".join(
        f"- model: `{model_name}`"
        for model_name in dict.fromkeys(summary_df["model_name"].astype(str).tolist())
    )
    if not model_lines:
        model_lines = "- Пусто"
    return f"""# Model Comparison Report

Дата: {created_at}
Protocol: `{protocol.name}`

## Что сравниваем
{model_lines}

## Источники benchmark
- host view: `{protocol.sources.host_view}`
- field view: `{protocol.sources.field_view}`
- features: `{", ".join(protocol.sources.feature_columns)}`
- split: `train/test`
- random_state: `{protocol.split.random_state}`
- test_size: `{protocol.split.test_size:.2f}`
- cv_folds: `{protocol.cv.n_splits}`
- cv_random_state: `{protocol.cv.random_state}`
- search_refit_metric: `{protocol.search.refit_metric}`
- precision@k: `{precision_k}`

## Итоговые метрики
{frame_to_text(summary_df)}

## Class-wise метрики
{frame_to_text(classwise_df)}

## Threshold Selection
{frame_to_text(normalized_thresholds_df)}

## Threshold-based Quality
{frame_to_text(normalized_quality_summary_df)}

## Class-wise Quality
{frame_to_text(normalized_quality_classwise_df)}

## Confusion Matrices
{frame_to_text(normalized_confusion_matrix_df)}

## Hyperparameter Search
{frame_to_text(normalized_search_summary_df)}

## Generalization Diagnostics
{frame_to_text(normalized_generalization_df)}

## Per-model Generalization Audit
{frame_to_text(normalized_audit_df)}

## Примечание
{note_text}
"""


def save_comparison_artifacts(
    run_name: str,
    scored_splits: list[ModelScoreFrames],
    *,
    output_dir: Path = DEFAULT_MODEL_COMPARISON_OUTPUT_DIR,
    precision_k: int = 50,
    search_summaries: Sequence[SearchSummaryEntry] = (),
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
    note: str = "",
) -> Path:
    """Сохранить markdown-, CSV- и scored-frame артефакты сравнения."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / run_name

    summary_df = build_comparison_summary_frame(
        scored_splits,
        precision_k=precision_k,
        protocol=protocol,
    )
    classwise_df = build_comparison_classwise_frame(
        scored_splits,
        precision_k=precision_k,
        protocol=protocol,
    )
    thresholds_df = build_comparison_thresholds_frame(
        scored_splits,
        protocol=protocol,
    )
    quality_summary_df = build_comparison_quality_summary_frame(
        scored_splits,
        protocol=protocol,
    )
    quality_classwise_df = build_comparison_quality_classwise_frame(
        scored_splits,
        protocol=protocol,
    )
    confusion_matrix_df = build_comparison_confusion_matrix_frame(
        scored_splits,
        protocol=protocol,
    )
    search_summary_df = build_search_summary_frame(search_summaries)
    generalization_df = build_generalization_diagnostics_frame(
        summary_df,
        search_summary_df=search_summary_df,
    )
    audit_df = build_model_generalization_audit_frame(
        summary_df,
        classwise_df,
        generalization_df,
    )
    markdown_path = prefix.with_suffix(".md")
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")
    classwise_path = prefix.with_name(f"{prefix.name}_classwise.csv")
    thresholds_path = prefix.with_name(f"{prefix.name}_thresholds.csv")
    quality_summary_path = prefix.with_name(f"{prefix.name}_quality_summary.csv")
    quality_classwise_path = prefix.with_name(f"{prefix.name}_quality_classwise.csv")
    confusion_matrix_path = prefix.with_name(f"{prefix.name}_confusion_matrices.csv")
    search_summary_path = prefix.with_name(f"{prefix.name}_search_summary.csv")
    generalization_path = prefix.with_name(f"{prefix.name}_generalization.csv")
    audit_path = prefix.with_name(f"{prefix.name}_generalization_audit.csv")
    audit_markdown_path = prefix.with_name(f"{prefix.name}_generalization_audit.md")

    markdown_path.write_text(
        build_comparison_markdown(
            summary_df,
            classwise_df,
            thresholds_df=thresholds_df,
            quality_summary_df=quality_summary_df,
            quality_classwise_df=quality_classwise_df,
            confusion_matrix_df=confusion_matrix_df,
            search_summary_df=search_summary_df,
            generalization_df=generalization_df,
            audit_df=audit_df,
            protocol=protocol,
            precision_k=precision_k,
            note=note,
        ),
        encoding="utf-8",
    )
    summary_df.to_csv(summary_path, index=False)
    classwise_df.to_csv(classwise_path, index=False)
    thresholds_df.to_csv(thresholds_path, index=False)
    quality_summary_df.to_csv(quality_summary_path, index=False)
    quality_classwise_df.to_csv(quality_classwise_path, index=False)
    confusion_matrix_df.to_csv(confusion_matrix_path, index=False)
    search_summary_df.to_csv(search_summary_path, index=False)
    generalization_df.to_csv(generalization_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    audit_markdown_path.write_text(
        build_generalization_audit_markdown(
            audit_df,
            protocol=protocol,
            note=note,
        ),
        encoding="utf-8",
    )

    for scored_split in scored_splits:
        train_path = prefix.with_name(
            f"{prefix.name}_{scored_split.model_name}_train_scores.csv"
        )
        test_path = prefix.with_name(
            f"{prefix.name}_{scored_split.model_name}_test_scores.csv"
        )
        scored_split.train_scored_df.to_csv(train_path, index=False)
        scored_split.test_scored_df.to_csv(test_path, index=False)

    return markdown_path


__all__ = [
    "DEFAULT_MODEL_COMPARISON_OUTPUT_DIR",
    "aggregate_search_summary_by_model",
    "build_comparison_classwise_frame",
    "build_comparison_confusion_matrix_frame",
    "build_comparison_quality_classwise_frame",
    "build_comparison_quality_summary_frame",
    "build_generalization_diagnostics_frame",
    "build_generalization_audit_markdown",
    "build_model_generalization_audit_frame",
    "build_comparison_markdown",
    "build_search_summary_frame",
    "build_comparison_summary_frame",
    "build_comparison_thresholds_frame",
    "frame_to_text",
    "save_comparison_artifacts",
]
