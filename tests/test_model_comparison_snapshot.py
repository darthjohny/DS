"""Тесты для snapshot-layer comparison моделей."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from analysis.model_comparison.snapshot import (
    SnapshotComparisonResult,
    SnapshotModelRun,
    build_snapshot_summary_frame,
    save_snapshot_artifacts,
)

from priority_pipeline.branching import RouterBranchFrames


def make_priority_frame(model_name: str, final_scores: list[float]) -> pd.DataFrame:
    """Собрать минимальный snapshot priority frame для тестов."""
    return pd.DataFrame(
        [
            {
                "source_id": index + 1,
                "model_name": model_name,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "router_label": "K_dwarf",
                "final_score": score,
                "priority_tier": "HIGH" if score >= 0.55 else "LOW",
                "reason_code": "HOST_SCORING" if score > 0 else "HOT_STAR",
            }
            for index, score in enumerate(final_scores)
        ]
    )


def test_build_snapshot_summary_frame_counts_rows_by_tier() -> None:
    """Snapshot summary должен корректно собирать tier-counts по моделям."""
    result = SnapshotComparisonResult(
        source_name="public.gaia_dr3_training",
        input_rows=3,
        router_df=pd.DataFrame([{"source_id": 1}, {"source_id": 2}, {"source_id": 3}]),
        branches=RouterBranchFrames(
            host_df=pd.DataFrame([{"source_id": 1}]),
            low_known_df=pd.DataFrame([{"source_id": 2}]),
            unknown_df=pd.DataFrame([{"source_id": 3}]),
        ),
        model_runs=[
            SnapshotModelRun(
                model_name="main_contrastive_v1",
                priority_df=make_priority_frame("main_contrastive_v1", [0.8, 0.0, 0.0]),
                top_df=pd.DataFrame(),
            ),
            SnapshotModelRun(
                model_name="baseline_random_forest",
                priority_df=make_priority_frame("baseline_random_forest", [0.9, 0.0, 0.0]),
                top_df=pd.DataFrame(),
            ),
        ],
    )

    summary_df = build_snapshot_summary_frame(result)

    assert summary_df["model_name"].tolist() == [
        "main_contrastive_v1",
        "baseline_random_forest",
    ]
    assert summary_df["router_rows"].eq(3).all()
    assert summary_df["host_candidates"].eq(1).all()
    assert summary_df["unknown_rows"].eq(1).all()
    assert summary_df["high_rows"].eq(1).all()
    assert summary_df["low_rows"].eq(2).all()


def test_save_snapshot_artifacts_writes_markdown_and_csv(tmp_path: Path) -> None:
    """Snapshot save должен создавать markdown и CSV артефакты."""
    result = SnapshotComparisonResult(
        source_name="public.gaia_dr3_training",
        input_rows=2,
        router_df=pd.DataFrame([{"source_id": 1}, {"source_id": 2}]),
        branches=RouterBranchFrames(
            host_df=pd.DataFrame([{"source_id": 1}]),
            low_known_df=pd.DataFrame([{"source_id": 2}]),
            unknown_df=pd.DataFrame(),
        ),
        model_runs=[
            SnapshotModelRun(
                model_name="main_contrastive_v1",
                priority_df=make_priority_frame("main_contrastive_v1", [0.7, 0.0]),
                top_df=make_priority_frame("main_contrastive_v1", [0.7]),
            )
        ],
    )

    markdown_path = save_snapshot_artifacts(
        "snapshot_smoke",
        result,
        output_dir=tmp_path,
        top_k=5,
        note="snapshot smoke test",
    )

    assert markdown_path.exists()
    assert (tmp_path / "snapshot_smoke_snapshot_summary.csv").exists()
    assert (tmp_path / "snapshot_smoke_snapshot_router.csv").exists()
    assert (
        tmp_path / "snapshot_smoke_snapshot_main_contrastive_v1_priority.csv"
    ).exists()
    assert (
        tmp_path / "snapshot_smoke_snapshot_main_contrastive_v1_top.csv"
    ).exists()
