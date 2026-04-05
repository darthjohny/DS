# Файл `benchmark_report.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from exohost.reporting.markdown_tables import format_scalar_line, frame_to_code_block

BENCHMARK_REPORT_FILENAME = "summary.md"


def load_benchmark_artifacts(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # Загружаем benchmark-таблицы и metadata из одного run_dir.
    benchmark_dir = Path(run_dir)
    metrics_df = pd.read_csv(benchmark_dir / "metrics.csv")
    cv_summary_df = pd.read_csv(benchmark_dir / "cv_summary.csv")
    metadata = json.loads((benchmark_dir / "metadata.json").read_text(encoding="utf-8"))
    return metrics_df, cv_summary_df, metadata


def build_benchmark_summary_lines(
    metrics_df: pd.DataFrame,
    cv_summary_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[str]:
    # Собираем краткое summary benchmark-прогона.
    summary_lines = [
        format_scalar_line("task_name", metadata.get("task_name", "unknown")),
        format_scalar_line("created_at_utc", metadata.get("created_at_utc", "unknown")),
        format_scalar_line("n_rows_full", metadata.get("n_rows_full", "unknown")),
        format_scalar_line("n_rows_train", metadata.get("n_rows_train", "unknown")),
        format_scalar_line("n_rows_test", metadata.get("n_rows_test", "unknown")),
    ]

    test_metrics = metrics_df.loc[metrics_df["split_name"] == "test"].copy()
    if not test_metrics.empty:
        best_test_row = test_metrics.sort_values(
            ["accuracy", "macro_f1", "balanced_accuracy"],
            ascending=[False, False, False],
            kind="mergesort",
            ignore_index=True,
        ).iloc[0]
        summary_lines.append(
            format_scalar_line(
                "best_test_model",
                f"{best_test_row['model_name']} (accuracy={best_test_row['accuracy']:.4f})",
            )
        )

    if not cv_summary_df.empty:
        best_cv_row = cv_summary_df.sort_values(
            ["mean_macro_f1", "mean_accuracy", "mean_balanced_accuracy"],
            ascending=[False, False, False],
            kind="mergesort",
            ignore_index=True,
        ).iloc[0]
        summary_lines.append(
            format_scalar_line(
                "best_cv_model",
                f"{best_cv_row['model_name']} (mean_macro_f1={best_cv_row['mean_macro_f1']:.4f})",
            )
        )

    return summary_lines


def build_benchmark_report_markdown(
    metrics_df: pd.DataFrame,
    cv_summary_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    # Собираем markdown-отчет по benchmark-прогону.
    test_metrics_df = metrics_df.loc[metrics_df["split_name"] == "test"].copy()
    test_metrics_df = test_metrics_df.sort_values(
        ["accuracy", "macro_f1", "balanced_accuracy"],
        ascending=[False, False, False],
        kind="mergesort",
        ignore_index=True,
    )
    sorted_cv_df = cv_summary_df.sort_values(
        ["mean_macro_f1", "mean_accuracy", "mean_balanced_accuracy"],
        ascending=[False, False, False],
        kind="mergesort",
        ignore_index=True,
    )

    sections = [
        "# Benchmark Report",
        "",
        *build_benchmark_summary_lines(metrics_df, cv_summary_df, metadata),
        "",
        "## Test Metrics",
        frame_to_code_block(test_metrics_df),
        "",
        "## Cross Validation",
        frame_to_code_block(sorted_cv_df),
    ]
    return "\n".join(sections).strip() + "\n"


def save_benchmark_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    # Сохраняем markdown-отчет рядом с benchmark-артефактами.
    benchmark_dir = Path(run_dir)
    metrics_df, cv_summary_df, metadata = load_benchmark_artifacts(benchmark_dir)
    report_text = build_benchmark_report_markdown(
        metrics_df,
        cv_summary_df,
        metadata,
    )
    report_path = Path(output_path) if output_path is not None else benchmark_dir / BENCHMARK_REPORT_FILENAME
    report_path.write_text(report_text, encoding="utf-8")
    return report_path
