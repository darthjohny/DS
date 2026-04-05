# Тестовый файл `test_benchmark_report.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from exohost.evaluation.split import DatasetSplit
from exohost.reporting.benchmark_artifacts import save_benchmark_artifacts
from exohost.reporting.benchmark_report import save_benchmark_report
from exohost.training.benchmark_runner import BenchmarkRunResult


def build_benchmark_result() -> BenchmarkRunResult:
    # Небольшой synthetic benchmark-результат для report-слоя.
    split = DatasetSplit(
        full_df=pd.DataFrame({"source_id": [1, 2, 3]}),
        train_df=pd.DataFrame({"source_id": [1, 2]}),
        test_df=pd.DataFrame({"source_id": [3]}),
    )
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "gmm_classifier",
                "split_name": "test",
                "n_rows": 1,
                "n_classes": 2,
                "accuracy": 0.80,
                "balanced_accuracy": 0.79,
                "macro_precision": 0.81,
                "macro_recall": 0.80,
                "macro_f1": 0.80,
                "roc_auc_ovr": 0.90,
            },
            {
                "model_name": "hist_gradient_boosting",
                "split_name": "test",
                "n_rows": 1,
                "n_classes": 2,
                "accuracy": 0.95,
                "balanced_accuracy": 0.95,
                "macro_precision": 0.95,
                "macro_recall": 0.95,
                "macro_f1": 0.95,
                "roc_auc_ovr": 0.98,
            },
        ]
    )
    cv_summary_df = pd.DataFrame(
        [
            {
                "model_name": "gmm_classifier",
                "cv_folds": 10,
                "mean_accuracy": 0.82,
                "mean_balanced_accuracy": 0.81,
                "mean_macro_f1": 0.82,
            },
            {
                "model_name": "hist_gradient_boosting",
                "cv_folds": 10,
                "mean_accuracy": 0.96,
                "mean_balanced_accuracy": 0.96,
                "mean_macro_f1": 0.96,
            },
        ]
    )
    target_distribution_df = pd.DataFrame(
        [
            {
                "split_name": "full",
                "target_label": "G",
                "n_rows": 3,
                "share": 1.0,
            }
        ]
    )
    return BenchmarkRunResult(
        task_name="spectral_class_classification",
        split=split,
        metrics_df=metrics_df,
        cv_summary_df=cv_summary_df,
        target_distribution_df=target_distribution_df,
    )


def test_save_benchmark_report_creates_summary_markdown(tmp_path: Path) -> None:
    # Проверяем сохранение markdown-summary рядом с benchmark-артефактами.
    result = build_benchmark_result()
    run_paths = save_benchmark_artifacts(
        result,
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 10, 0, 0, tzinfo=UTC),
    )

    report_path = save_benchmark_report(run_paths.run_dir)
    report_text = report_path.read_text(encoding="utf-8")

    assert report_path.name == "summary.md"
    assert "# Benchmark Report" in report_text
    assert "best_test_model" in report_text
    assert "hist_gradient_boosting" in report_text
