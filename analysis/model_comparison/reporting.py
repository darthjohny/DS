"""Сборка markdown- и CSV-артефактов comparison-layer."""

from __future__ import annotations

import json
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
from analysis.model_comparison.metrics import (
    build_classwise_metrics_frame,
    build_metrics_frame,
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


def build_comparison_markdown(
    summary_df: pd.DataFrame,
    classwise_df: pd.DataFrame,
    *,
    search_summary_df: pd.DataFrame | None = None,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
    precision_k: int = 50,
    note: str = "",
) -> str:
    """Собрать markdown summary для comparative benchmark."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    note_text = note.strip() or "-"
    normalized_search_summary_df = (
        pd.DataFrame() if search_summary_df is None else search_summary_df
    )
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

## Hyperparameter Search
{frame_to_text(normalized_search_summary_df)}

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
    search_summary_df = build_search_summary_frame(search_summaries)
    markdown_path = prefix.with_suffix(".md")
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")
    classwise_path = prefix.with_name(f"{prefix.name}_classwise.csv")
    search_summary_path = prefix.with_name(f"{prefix.name}_search_summary.csv")

    markdown_path.write_text(
        build_comparison_markdown(
            summary_df,
            classwise_df,
            search_summary_df=search_summary_df,
            protocol=protocol,
            precision_k=precision_k,
            note=note,
        ),
        encoding="utf-8",
    )
    summary_df.to_csv(summary_path, index=False)
    classwise_df.to_csv(classwise_path, index=False)
    search_summary_df.to_csv(search_summary_path, index=False)

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
    "build_comparison_classwise_frame",
    "build_comparison_markdown",
    "build_search_summary_frame",
    "build_comparison_summary_frame",
    "frame_to_text",
    "save_comparison_artifacts",
]
