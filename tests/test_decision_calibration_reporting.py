"""Тесты reporting-контура офлайн-калибровки decision layer."""

from __future__ import annotations

import pandas as pd

from decision_calibration import (
    BaseScoringResult,
    CalibrationConfig,
    ReadyDatasetRecord,
    build_iteration_summary,
    save_iteration_artifacts,
    top_candidates_frame,
)


def test_build_iteration_summary_tracks_unknown_rows_and_share() -> None:
    """Summary должен отдельно считать unknown-ветку и её долю."""
    dataset = ReadyDatasetRecord(
        relation_name="lab.ready_dataset",
        source_name="lab.source_dataset",
        status="READY",
        row_count=4,
        validated_at=None,
    )
    base_result = BaseScoringResult(
        input_df=pd.DataFrame({"source_id": [1, 2, 3, 4]}),
        router_df=pd.DataFrame({"source_id": [1, 2, 3, 4]}),
        host_input_df=pd.DataFrame({"source_id": [1]}),
        low_known_input_df=pd.DataFrame({"source_id": [2, 3]}),
        unknown_input_df=pd.DataFrame({"source_id": [4]}),
        low_input_df=pd.DataFrame({"source_id": [2, 3, 4]}),
        host_scored_df=pd.DataFrame({"source_id": [1]}),
    )
    ordered_results = pd.DataFrame({"final_score": [0.8, 0.0, 0.0, 0.0]})

    summary = build_iteration_summary(
        run_id="run_1",
        dataset=dataset,
        base_result=base_result,
        ordered_results=ordered_results,
        top_n=10,
        router_score_mode="gaussian_log_posterior_v1",
        host_score_mode="host_vs_field_log_lr_v1",
        host_model_version_value="gaussian_host_field_v1",
    )

    assert summary.host_rows == 1
    assert summary.low_known_rows == 2
    assert summary.unknown_rows == 1
    assert summary.low_rows == 3
    assert summary.unknown_share == 0.25


def test_top_candidates_frame_excludes_unknown_rows() -> None:
    """Top-N таблица не должна включать `UNKNOWN`, даже если он есть в ranking."""
    ordered_results = pd.DataFrame(
        [
            {
                "source_id": 10,
                "predicted_spec_class": "UNKNOWN",
                "predicted_evolution_stage": "unknown",
                "final_score": 0.95,
                "priority_tier": "LOW",
                "reason_code": "ROUTER_UNKNOWN",
            },
            {
                "source_id": 20,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "final_score": 0.80,
                "priority_tier": "HIGH",
                "reason_code": "HOST_SCORING",
            },
        ]
    )

    top_candidates = top_candidates_frame(ordered_results, top_n=5)

    assert top_candidates["source_id"].tolist() == [20]
    assert top_candidates["predicted_spec_class"].tolist() == ["K"]


def test_save_iteration_artifacts_persists_unknown_aware_report_and_tables(
    tmp_path,
) -> None:
    """Артефакты итерации должны сохранять unknown-aware summary и top без `UNKNOWN`."""
    dataset = ReadyDatasetRecord(
        relation_name="lab.ready_dataset",
        source_name="lab.source_dataset",
        status="READY",
        row_count=4,
        validated_at=None,
    )
    base_result = BaseScoringResult(
        input_df=pd.DataFrame({"source_id": [1, 2, 3, 4]}),
        router_df=pd.DataFrame({"source_id": [1, 2, 3, 4]}),
        host_input_df=pd.DataFrame({"source_id": [2]}),
        low_known_input_df=pd.DataFrame({"source_id": [3]}),
        unknown_input_df=pd.DataFrame({"source_id": [1, 4]}),
        low_input_df=pd.DataFrame({"source_id": [1, 3, 4]}),
        host_scored_df=pd.DataFrame({"source_id": [2]}),
    )
    ordered_results = pd.DataFrame(
        [
            {
                "source_id": 1,
                "predicted_spec_class": "UNKNOWN",
                "predicted_evolution_stage": "unknown",
                "final_score": 0.95,
                "priority_tier": "LOW",
                "reason_code": "ROUTER_UNKNOWN",
            },
            {
                "source_id": 2,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "final_score": 0.80,
                "priority_tier": "HIGH",
                "reason_code": "HOST_SCORING",
            },
        ]
    )
    summary = build_iteration_summary(
        run_id="run_7",
        dataset=dataset,
        base_result=base_result,
        ordered_results=ordered_results,
        top_n=5,
        router_score_mode="gaussian_log_posterior_v1",
        host_score_mode="host_vs_field_log_lr_v1",
        host_model_version_value="gaussian_host_field_v1",
    )

    markdown_path = save_iteration_artifacts(
        logbook_dir=tmp_path,
        config=CalibrationConfig(),
        summary=summary,
        ordered_results=ordered_results,
        top_n=5,
        iteration_note="unknown branch accounted for",
        next_iteration_number=7,
    )

    markdown_text = markdown_path.read_text(encoding="utf-8")
    top_candidates = pd.read_csv(tmp_path / "iteration_007_top_candidates.csv")
    class_distribution = pd.read_csv(tmp_path / "iteration_007_class_distribution.csv")
    score_summary = pd.read_csv(tmp_path / "iteration_007_score_summary.csv")

    assert markdown_path.name == "iteration_007.md"
    assert "low_known_rows: 1" in markdown_text
    assert "unknown_rows: 2" in markdown_text
    assert "unknown_share: 0.5000" in markdown_text
    assert "unknown branch accounted for" in markdown_text
    assert top_candidates["source_id"].tolist() == [2]
    assert top_candidates["predicted_spec_class"].tolist() == ["K"]
    assert class_distribution["predicted_spec_class"].tolist() == ["K"]
    assert score_summary["metric"].tolist()[:4] == [
        "run_id",
        "relation_name",
        "source_name",
        "input_rows",
    ]
