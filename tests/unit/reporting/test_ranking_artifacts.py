# Тестовый файл `test_ranking_artifacts.py` домена `reporting`.
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

from exohost.reporting.ranking_artifacts import (
    build_ranking_artifact_paths,
    save_ranking_artifacts,
)


def build_ranking_frame() -> pd.DataFrame:
    # Небольшая synthetic ranking-таблица для проверки файлового слоя.
    return pd.DataFrame(
        [
            {
                "source_id": "1",
                "priority_score": 0.91,
                "priority_label": "high",
            },
            {
                "source_id": "2",
                "priority_score": 0.52,
                "priority_label": "medium",
            },
        ]
    )


def test_build_ranking_artifact_paths_creates_expected_layout(tmp_path: Path) -> None:
    # Проверяем стандартные имена файлов ranking-прогона.
    now = datetime(2026, 3, 20, 9, 50, 0, tzinfo=UTC)

    paths = build_ranking_artifact_paths(
        output_dir=tmp_path,
        ranking_name="router_candidates",
        now=now,
    )

    assert paths.run_dir.parent == tmp_path
    assert paths.ranking_csv_path.name == "ranking.csv"
    assert paths.metadata_json_path.name == "metadata.json"
    assert "router_candidates" in paths.run_dir.name


def test_save_ranking_artifacts_writes_table_and_metadata(tmp_path: Path) -> None:
    # Проверяем сохранение ranking-таблицы и metadata.
    now = datetime(2026, 3, 20, 9, 50, 0, tzinfo=UTC)
    ranking_df = build_ranking_frame()

    paths = save_ranking_artifacts(
        ranking_df,
        ranking_name="router_candidates",
        output_dir=tmp_path,
        now=now,
        extra_metadata={"input_csv": "candidates.csv"},
    )

    saved_frame = pd.read_csv(paths.ranking_csv_path)
    metadata = json.loads(paths.metadata_json_path.read_text(encoding="utf-8"))

    assert saved_frame.shape == (2, 3)
    assert metadata["ranking_name"] == "router_candidates"
    assert metadata["n_rows"] == 2
    assert metadata["priority_label_distribution"]["high"] == 1
    assert metadata["context"]["input_csv"] == "candidates.csv"
