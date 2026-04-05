# Тестовый файл `test_final_decision_artifacts.py` домена `reporting`.
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

from exohost.reporting.final_decision_artifacts import (
    build_final_decision_artifact_paths,
    load_final_decision_artifacts,
    save_final_decision_artifacts,
)


def build_decision_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{"source_id": 1, "quality_state": "pass", "ood_decision": "in_domain"}]
    )


def build_final_decision_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "final_domain_state": "id",
                "final_quality_state": "pass",
                "final_coarse_class": "G",
                "priority_state": "high",
            }
        ]
    )


def build_priority_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{"source_id": 1, "spec_class": "G", "host_similarity_score": 0.92}]
    )


def build_priority_ranking_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "priority_score": 0.81,
                "priority_label": "high",
                "priority_reason": "strong host-like signal",
            }
        ]
    )


def test_build_final_decision_artifact_paths_creates_stable_layout(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 3, 28, 20, 0, 0, tzinfo=UTC)
    paths = build_final_decision_artifact_paths(
        output_dir=tmp_path,
        pipeline_name="hierarchical_final_decision",
        now=now,
    )

    assert paths.run_dir.parent == tmp_path
    assert paths.decision_input_csv_path.name == "decision_input.csv"
    assert paths.final_decision_csv_path.name == "final_decision.csv"
    assert paths.priority_input_csv_path.name == "priority_input.csv"
    assert paths.priority_ranking_csv_path.name == "priority_ranking.csv"
    assert paths.metadata_json_path.name == "metadata.json"


def test_save_and_load_final_decision_artifacts_roundtrip(tmp_path: Path) -> None:
    paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=build_decision_input_df(),
        final_decision_df=build_final_decision_df(),
        priority_input_df=build_priority_input_df(),
        priority_ranking_df=build_priority_ranking_df(),
        output_dir=tmp_path,
        extra_metadata={"kind": "decision_run"},
    )

    loaded_bundle = load_final_decision_artifacts(paths.run_dir)

    assert loaded_bundle.metadata["pipeline_name"] == "hierarchical_final_decision"
    assert loaded_bundle.metadata["context"]["kind"] == "decision_run"
    assert list(loaded_bundle.final_decision_df["final_domain_state"]) == ["id"]
    assert list(loaded_bundle.priority_ranking_df["priority_label"]) == ["high"]
