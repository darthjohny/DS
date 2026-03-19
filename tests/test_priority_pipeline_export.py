"""Тесты dedicated export-layer production priority pipeline."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import cast

import pandas as pd
from sqlalchemy.engine import Engine

import priority_pipeline.export as export_mod
from priority_pipeline.contracts import PipelineRunResult
from priority_pipeline.export import (
    OPERATIONAL_SHORTLIST_COLUMNS,
    build_operational_shortlist,
    build_shortlist_summary_frame,
    save_operational_artifacts,
)


def make_priority_results_df() -> pd.DataFrame:
    """Собрать минимальный priority frame для operational export smoke tests."""
    return pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "source_id": 101,
                "ra": 12.34,
                "dec": -5.67,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "gauss_label": "K",
                "host_posterior": 0.91,
                "final_score": 0.74,
                "priority_tier": "HIGH",
                "reason_code": "HOST_SCORING",
            },
            {
                "run_id": "run_1",
                "source_id": 202,
                "ra": 22.22,
                "dec": -1.23,
                "predicted_spec_class": "M",
                "predicted_evolution_stage": "dwarf",
                "gauss_label": None,
                "host_posterior": 0.83,
                "final_score": 0.68,
                "priority_tier": "HIGH",
                "reason_code": "HOST_SCORING",
            },
            {
                "run_id": "run_1",
                "source_id": 303,
                "ra": 33.33,
                "dec": -2.34,
                "predicted_spec_class": "G",
                "predicted_evolution_stage": "dwarf",
                "gauss_label": "G",
                "host_posterior": 0.78,
                "final_score": 0.44,
                "priority_tier": "MEDIUM",
                "reason_code": "HOST_SCORING",
            },
            {
                "run_id": "run_1",
                "source_id": 404,
                "ra": 44.44,
                "dec": -3.45,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "giant",
                "gauss_label": "K",
                "host_posterior": 0.88,
                "final_score": 0.72,
                "priority_tier": "HIGH",
                "reason_code": "EVOLVED_STAR",
            },
            {
                "run_id": "run_1",
                "source_id": 505,
                "ra": 55.55,
                "dec": -4.56,
                "predicted_spec_class": "A",
                "predicted_evolution_stage": "dwarf",
                "gauss_label": "A",
                "host_posterior": 0.77,
                "final_score": 0.66,
                "priority_tier": "HIGH",
                "reason_code": "HOT_STAR",
            },
        ]
    )


def test_build_operational_shortlist_filters_runtime_results_by_v1_rules() -> None:
    """Shortlist должен брать только HIGH dwarf-кандидатов K/M/G из runtime."""
    shortlist_df = build_operational_shortlist(make_priority_results_df())

    assert shortlist_df.columns.tolist() == list(OPERATIONAL_SHORTLIST_COLUMNS)
    assert shortlist_df["source_id"].tolist() == [101, 202]
    assert shortlist_df["observation_priority"].tolist() == [1, 2]
    assert shortlist_df["rank_in_priority"].tolist() == [1, 1]
    assert shortlist_df["host_like_percent"].tolist() == [91.0, 83.0]
    assert shortlist_df["host_like_profile"].tolist() == ["K", "-"]


def test_build_operational_shortlist_keeps_canonical_schema_when_no_rows_match() -> None:
    """Даже пустой shortlist после фильтрации должен иметь канонические колонки."""
    df_priority = make_priority_results_df().assign(priority_tier="LOW")

    shortlist_df = build_operational_shortlist(df_priority)

    assert shortlist_df.empty
    assert shortlist_df.columns.tolist() == list(OPERATIONAL_SHORTLIST_COLUMNS)


def test_save_operational_artifacts_writes_versioned_csv_and_markdown(
    tmp_path: Path,
) -> None:
    """Export layer должен сохранять router/priority/summary/shortlist артефакты."""
    result = PipelineRunResult(
        run_id="run_2026_03_19",
        router_results=pd.DataFrame(
            [
                {
                    "run_id": "run_2026_03_19",
                    "source_id": 101,
                    "predicted_spec_class": "K",
                }
            ]
        ),
        priority_results=make_priority_results_df(),
    )

    markdown_path = save_operational_artifacts(
        run_name="production_priority_smoke",
        input_source="public.gaia_dr3_training",
        limit=5000,
        result=result,
        output_dir=tmp_path,
        note="operational export smoke test",
    )

    assert markdown_path == tmp_path / "production_priority_smoke.md"
    assert markdown_path.exists()
    assert (tmp_path / "production_priority_smoke_router.csv").exists()
    assert (tmp_path / "production_priority_smoke_priority.csv").exists()
    assert (tmp_path / "production_priority_smoke_tier_summary.csv").exists()
    assert (tmp_path / "production_priority_smoke_class_summary.csv").exists()
    assert (tmp_path / "production_priority_smoke_shortlist.csv").exists()
    assert (tmp_path / "production_priority_smoke_shortlist_summary.csv").exists()

    shortlist_df = pd.read_csv(
        tmp_path / "production_priority_smoke_shortlist.csv",
        low_memory=False,
    )
    shortlist_summary_df = pd.read_csv(
        tmp_path / "production_priority_smoke_shortlist_summary.csv",
        low_memory=False,
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert shortlist_df["source_id"].tolist() == [101, 202]
    assert shortlist_summary_df["observation_priority"].tolist() == [1, 2]
    assert shortlist_summary_df["n_rows"].tolist() == [1, 1]
    assert "run_id: `run_2026_03_19`" in markdown
    assert "operational export smoke test" in markdown


def test_build_shortlist_summary_frame_counts_rows_by_priority() -> None:
    """Shortlist summary должен честно агрегировать по physical priority."""
    shortlist_df = build_operational_shortlist(make_priority_results_df())

    summary_df = build_shortlist_summary_frame(shortlist_df)

    assert summary_df.to_dict(orient="records") == [
        {"observation_priority": 1, "n_rows": 1},
        {"observation_priority": 2, "n_rows": 1},
    ]


def test_export_main_orchestrates_pipeline_and_artifact_save(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """CLI main должен связать parse_args, pipeline run и artifact save."""
    result = PipelineRunResult(
        run_id="run_main_smoke",
        router_results=pd.DataFrame([{"source_id": 1}]),
        priority_results=make_priority_results_df(),
    )
    captured: dict[str, object] = {}

    def fake_parse_args() -> Namespace:
        return Namespace(
            run_name=None,
            input_source="public.gaia_dr3_training",
            limit=5000,
            output_dir=tmp_path,
            persist=False,
            note="main smoke",
        )

    def fake_make_engine_from_env() -> Engine:
        captured["engine_built"] = True
        return cast(Engine, object())

    def fake_run_pipeline(
        *,
        engine: Engine,
        input_source: str,
        limit: int | None,
        persist: bool,
    ) -> PipelineRunResult:
        captured["engine"] = engine
        captured["input_source"] = input_source
        captured["limit"] = limit
        captured["persist"] = persist
        return result

    def fake_save_operational_artifacts(
        *,
        run_name: str,
        input_source: str,
        limit: int | None,
        result: PipelineRunResult,
        output_dir: Path,
        note: str,
    ) -> Path:
        captured["run_name"] = run_name
        captured["saved_input_source"] = input_source
        captured["saved_limit"] = limit
        captured["saved_result"] = result
        captured["output_dir"] = output_dir
        captured["note"] = note
        return output_dir / f"{run_name}.md"

    monkeypatch.setattr(export_mod, "parse_args", fake_parse_args)
    monkeypatch.setattr(export_mod, "make_engine_from_env", fake_make_engine_from_env)
    monkeypatch.setattr(export_mod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        export_mod,
        "save_operational_artifacts",
        fake_save_operational_artifacts,
    )

    export_mod.main()

    assert captured["engine_built"] is True
    assert captured["input_source"] == "public.gaia_dr3_training"
    assert captured["limit"] == 5000
    assert captured["persist"] is False
    assert str(captured["run_name"]).startswith(
        "production_priority_"
    )
    assert captured["saved_result"] is result
    assert captured["output_dir"] == tmp_path
    assert captured["note"] == "main smoke"
