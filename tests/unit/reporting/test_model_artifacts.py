# Тестовый файл `test_model_artifacts.py` домена `reporting`.
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

import joblib
import pandas as pd

from exohost.evaluation.protocol import SPECTRAL_CLASS_CLASSIFICATION_TASK
from exohost.reporting.model_artifacts import (
    build_model_artifact_paths,
    load_model_artifact,
    load_model_artifact_metadata,
    save_model_artifacts,
)
from exohost.training.train_runner import TrainRunResult


def test_build_model_artifact_paths_creates_stable_layout(tmp_path: Path) -> None:
    # Проверяем имена файлов и каталога model artifacts.
    now = datetime(2026, 3, 20, 11, 0, 0, tzinfo=UTC)
    paths = build_model_artifact_paths(
        output_dir=tmp_path,
        task_name="spectral_class_classification",
        model_name="hist_gradient_boosting",
        now=now,
    )

    assert paths.run_dir.parent == tmp_path
    assert paths.model_joblib_path.name == "model.joblib"
    assert paths.label_distribution_csv_path.name == "label_distribution.csv"
    assert paths.metadata_json_path.name == "metadata.json"


def test_save_model_artifacts_writes_joblib_and_metadata(
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Проверяем сохранение model artifact, label distribution и metadata.
    result = small_spectral_class_train_result
    now = datetime(2026, 3, 20, 11, 0, 0, tzinfo=UTC)

    paths = save_model_artifacts(
        result,
        output_dir=tmp_path,
        now=now,
        extra_metadata={"task": result.task_name},
    )

    saved_estimator = joblib.load(paths.model_joblib_path)
    label_distribution_df = pd.read_csv(paths.label_distribution_csv_path)
    metadata = json.loads(paths.metadata_json_path.read_text(encoding="utf-8"))

    assert saved_estimator.model_name == "hist_gradient_boosting"
    assert set(label_distribution_df["target_label"]) == {"G", "K"}
    assert metadata["task_name"] == "spectral_class_classification"
    assert metadata["model_name"] == "hist_gradient_boosting"
    assert metadata["context"]["task"] == "spectral_class_classification"


def test_load_model_artifact_restores_typed_contract(
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Проверяем, что при загрузке metadata сразу приводится к рабочему контракту.
    result = small_spectral_class_train_result
    paths = save_model_artifacts(result, output_dir=tmp_path)

    loaded_artifact = load_model_artifact(paths.run_dir)

    assert loaded_artifact.task_name == "spectral_class_classification"
    assert loaded_artifact.target_column == "spec_class"
    assert loaded_artifact.feature_columns == SPECTRAL_CLASS_CLASSIFICATION_TASK.feature_columns
    assert loaded_artifact.model_name == "hist_gradient_boosting"


def test_load_model_artifact_metadata_reads_json_without_joblib(
    tmp_path: Path,
    small_spectral_class_train_result: TrainRunResult,
) -> None:
    # Проверяем metadata-only path без materialize estimator.
    result = small_spectral_class_train_result
    paths = save_model_artifacts(result, output_dir=tmp_path)

    metadata = load_model_artifact_metadata(paths.run_dir)

    assert metadata["task_name"] == "spectral_class_classification"
    assert metadata["model_name"] == "hist_gradient_boosting"
