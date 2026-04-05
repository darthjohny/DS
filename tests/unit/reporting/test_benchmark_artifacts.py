# Тестовый файл `test_benchmark_artifacts.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from exohost.evaluation.split import DatasetSplit
from exohost.reporting.benchmark_artifacts import (
    build_benchmark_artifact_paths,
    save_benchmark_artifacts,
)
from exohost.training.benchmark_runner import BenchmarkRunResult


def build_benchmark_result() -> BenchmarkRunResult:
    # Небольшой synthetic benchmark-результат для файлового сохранения.
    full_df = pd.DataFrame({"source_id": [1, 2, 3]})
    train_df = pd.DataFrame({"source_id": [1, 2]})
    test_df = pd.DataFrame({"source_id": [3]})
    split = DatasetSplit(
        full_df=full_df,
        train_df=train_df,
        test_df=test_df,
    )
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "gmm_classifier",
                "split_name": "test",
                "accuracy": 0.9,
            }
        ]
    )
    cv_summary_df = pd.DataFrame(
        [
            {
                "model_name": "gmm_classifier",
                "cv_folds": 10,
                "mean_accuracy": 0.85,
            }
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


def test_build_benchmark_artifact_paths_creates_stable_layout(tmp_path: Path) -> None:
    # Проверяем имена файлов и каталога артефактов.
    now = datetime(2026, 3, 20, 9, 15, 30, tzinfo=UTC)
    paths = build_benchmark_artifact_paths(
        output_dir=tmp_path,
        task_name="host_field_classification",
        now=now,
    )

    assert paths.run_dir.parent == tmp_path
    assert paths.metrics_csv_path.name == "metrics.csv"
    assert paths.cv_summary_csv_path.name == "cv_summary.csv"
    assert paths.target_distribution_csv_path.name == "target_distribution.csv"
    assert paths.metadata_json_path.name == "metadata.json"
    assert "host_field_classification" in paths.run_dir.name


def test_save_benchmark_artifacts_writes_csv_and_metadata(tmp_path: Path) -> None:
    # Проверяем, что reporting-слой сохраняет таблицы и metadata на диск.
    now = datetime(2026, 3, 20, 9, 15, 30, tzinfo=UTC)
    result = build_benchmark_result()

    paths = save_benchmark_artifacts(
        result,
        output_dir=tmp_path,
        now=now,
        extra_metadata={"limit": 1000, "task": result.task_name},
    )

    assert paths.metrics_csv_path.exists()
    assert paths.cv_summary_csv_path.exists()
    assert paths.target_distribution_csv_path.exists()
    assert paths.metadata_json_path.exists()

    metrics_df = pd.read_csv(paths.metrics_csv_path)
    cv_summary_df = pd.read_csv(paths.cv_summary_csv_path)
    target_distribution_df = pd.read_csv(paths.target_distribution_csv_path)
    metadata = json.loads(paths.metadata_json_path.read_text(encoding="utf-8"))

    assert metrics_df.loc[0, "model_name"] == "gmm_classifier"
    assert cv_summary_df.loc[0, "cv_folds"] == 10
    assert target_distribution_df.loc[0, "split_name"] == "full"
    assert metadata["task_name"] == "spectral_class_classification"
    assert metadata["n_rows_full"] == 3
    assert metadata["context"]["limit"] == 1000
