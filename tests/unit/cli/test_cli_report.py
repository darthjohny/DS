# Тестовый файл `test_cli_report.py` домена `cli`.
#
# Этот файл проверяет только:
# - проверку логики домена: CLI-команды и их orchestration-сценарии;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `cli` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from exohost.cli.main import main
from exohost.evaluation.split import DatasetSplit
from exohost.models.inference import ModelScoringResult
from exohost.reporting.benchmark_artifacts import save_benchmark_artifacts
from exohost.reporting.ranking_artifacts import save_ranking_artifacts
from exohost.reporting.scoring_artifacts import save_scoring_artifacts
from exohost.training.benchmark_runner import BenchmarkRunResult


def build_benchmark_result() -> BenchmarkRunResult:
    # Небольшой synthetic benchmark-результат для CLI report.
    split = DatasetSplit(
        full_df=pd.DataFrame({"source_id": [1, 2, 3]}),
        train_df=pd.DataFrame({"source_id": [1, 2]}),
        test_df=pd.DataFrame({"source_id": [3]}),
    )
    return BenchmarkRunResult(
        task_name="spectral_class_classification",
        split=split,
        metrics_df=pd.DataFrame(
            [
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
                }
            ]
        ),
        cv_summary_df=pd.DataFrame(
            [
                {
                    "model_name": "hist_gradient_boosting",
                    "cv_folds": 10,
                    "mean_accuracy": 0.96,
                    "mean_balanced_accuracy": 0.96,
                    "mean_macro_f1": 0.96,
                }
            ]
        ),
        target_distribution_df=pd.DataFrame(
            [
                {
                    "split_name": "full",
                    "target_label": "G",
                    "n_rows": 3,
                    "share": 1.0,
                }
            ]
        ),
    )


def build_scoring_result() -> ModelScoringResult:
    # Небольшой synthetic scoring-результат для CLI report.
    return ModelScoringResult(
        task_name="host_field_classification",
        target_column="host_label",
        model_name="hist_gradient_boosting",
        n_rows=2,
        scored_df=pd.DataFrame(
            [
                {
                    "source_id": "1",
                    "predicted_host_label": "host",
                    "predicted_host_label_confidence": 0.95,
                    "probability__host": 0.95,
                    "probability__field": 0.05,
                    "host_similarity_score": 0.95,
                },
                {
                    "source_id": "2",
                    "predicted_host_label": "field",
                    "predicted_host_label_confidence": 0.70,
                    "probability__host": 0.30,
                    "probability__field": 0.70,
                    "host_similarity_score": 0.30,
                },
            ]
        ),
    )
def test_cli_report_builds_benchmark_summary(tmp_path: Path) -> None:
    # Проверяем report-команду для benchmark run_dir.
    run_paths = save_benchmark_artifacts(
        build_benchmark_result(),
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 10, 10, 0, tzinfo=UTC),
    )

    exit_code = main(
        [
            "report",
            "--kind",
            "benchmark",
            "--run-dir",
            str(run_paths.run_dir),
        ]
    )

    assert exit_code == 0
    assert (run_paths.run_dir / "summary.md").exists()


def test_cli_report_builds_ranking_summary(tmp_path: Path) -> None:
    # Проверяем report-команду для ranking run_dir.
    run_paths = save_ranking_artifacts(
        pd.DataFrame(
            [
                {
                    "source_id": "1",
                    "priority_score": 0.91,
                    "priority_label": "high",
                }
            ]
        ),
        ranking_name="router_candidates",
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 10, 12, 0, tzinfo=UTC),
    )

    exit_code = main(
        [
            "report",
            "--kind",
            "ranking",
            "--run-dir",
            str(run_paths.run_dir),
        ]
    )

    assert exit_code == 0
    assert (run_paths.run_dir / "summary.md").exists()


def test_cli_report_builds_scoring_summary(tmp_path: Path) -> None:
    # Проверяем report-команду для scoring run_dir.
    run_paths = save_scoring_artifacts(
        build_scoring_result(),
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 10, 14, 0, tzinfo=UTC),
    )

    exit_code = main(
        [
            "report",
            "--kind",
            "scoring",
            "--run-dir",
            str(run_paths.run_dir),
        ]
    )

    assert exit_code == 0
    assert (run_paths.run_dir / "summary.md").exists()
