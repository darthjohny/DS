# Тестовый файл `test_scoring_artifacts.py` домена `reporting`.
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

from exohost.models.inference import ModelScoringResult
from exohost.reporting.scoring_artifacts import (
    build_scoring_artifact_paths,
    load_scoring_artifacts,
    save_scoring_artifacts,
)


def build_scoring_result() -> ModelScoringResult:
    # Собираем маленький scoring-result для проверки artifact-слоя.
    return ModelScoringResult(
        task_name="spectral_class_classification",
        target_column="spec_class",
        model_name="hist_gradient_boosting",
        n_rows=2,
        scored_df=pd.DataFrame(
            [
                {
                    "source_id": "1",
                    "predicted_spec_class": "G",
                    "predicted_spec_class_confidence": 0.91,
                },
                {
                    "source_id": "2",
                    "predicted_spec_class": "K",
                    "predicted_spec_class_confidence": 0.88,
                },
            ]
        ),
    )


def test_build_scoring_artifact_paths_creates_stable_layout(tmp_path: Path) -> None:
    # Проверяем имена файлов и каталога scoring-артефактов.
    now = datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC)
    paths = build_scoring_artifact_paths(
        output_dir=tmp_path,
        task_name="spectral_class_classification",
        model_name="hist_gradient_boosting",
        now=now,
    )

    assert paths.run_dir.parent == tmp_path
    assert paths.scored_csv_path.name == "scored.csv"
    assert paths.metadata_json_path.name == "metadata.json"


def test_save_scoring_artifacts_writes_csv_and_metadata(tmp_path: Path) -> None:
    # Проверяем сохранение scored frame и metadata.
    result = build_scoring_result()
    now = datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC)

    paths = save_scoring_artifacts(
        result,
        output_dir=tmp_path,
        now=now,
        extra_metadata={"score_mode": "model_scoring"},
    )

    scored_df = pd.read_csv(paths.scored_csv_path)
    metadata = json.loads(paths.metadata_json_path.read_text(encoding="utf-8"))

    assert list(scored_df["predicted_spec_class"]) == ["G", "K"]
    assert metadata["task_name"] == "spectral_class_classification"
    assert metadata["model_name"] == "hist_gradient_boosting"
    assert metadata["context"]["score_mode"] == "model_scoring"


def test_load_scoring_artifacts_reads_saved_csv_and_metadata(tmp_path: Path) -> None:
    # Проверяем обратную загрузку scoring-артефактов из run_dir.
    result = build_scoring_result()
    paths = save_scoring_artifacts(result, output_dir=tmp_path)

    scored_df, metadata = load_scoring_artifacts(paths.run_dir)

    assert list(scored_df["predicted_spec_class"]) == ["G", "K"]
    assert metadata["task_name"] == "spectral_class_classification"
