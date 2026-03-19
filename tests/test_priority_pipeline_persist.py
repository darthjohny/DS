"""Точечные тесты persist- и relation-helper-ов production pipeline."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest
from sqlalchemy.engine import Engine

import priority_pipeline.persist as persist
from priority_pipeline.relations import split_relation_name


def test_build_persist_payload_keeps_canonical_order_and_drops_extras() -> None:
    """Persist payload должен брать только совместимые колонки в нужном порядке."""
    df = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "source_id": 101,
                "final_score": 0.72,
                "priority_tier": "HIGH",
                "reason_code": "HOST_SCORING",
                "extra_debug": "drop-me",
            }
        ]
    )

    payload = persist.build_persist_payload(
        df=df,
        ordered_columns=(
            "run_id",
            "source_id",
            "final_score",
            "priority_tier",
            "reason_code",
        ),
        available_columns=[
            "source_id",
            "run_id",
            "priority_tier",
            "final_score",
            "reason_code",
            "db_only_column",
        ],
        required_columns=("run_id", "source_id", "final_score"),
        table_name="lab.gaia_priority_results",
    )

    assert payload.columns.tolist() == [
        "run_id",
        "source_id",
        "final_score",
        "priority_tier",
        "reason_code",
    ]
    assert payload.iloc[0].to_dict() == {
        "run_id": "run_1",
        "source_id": 101,
        "final_score": 0.72,
        "priority_tier": "HIGH",
        "reason_code": "HOST_SCORING",
    }


def test_build_persist_payload_fails_when_relation_misses_required_columns() -> None:
    """Persist helper должен явно падать, если обязательных колонок нет в relation."""
    df = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "source_id": 101,
                "final_score": 0.72,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="missing required columns: final_score"):
        persist.build_persist_payload(
            df=df,
            ordered_columns=("run_id", "source_id", "final_score"),
            available_columns=["run_id", "source_id"],
            required_columns=("run_id", "source_id", "final_score"),
            table_name="lab.gaia_priority_results",
        )


def test_build_persist_payload_fails_when_frame_misses_required_columns() -> None:
    """Persist helper должен явно падать, если обязательных колонок нет в DataFrame."""
    df = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "source_id": 101,
                "priority_tier": "HIGH",
            }
        ]
    )

    with pytest.raises(RuntimeError, match="missing required persist columns: final_score"):
        persist.build_persist_payload(
            df=df,
            ordered_columns=("run_id", "source_id", "final_score", "priority_tier"),
            available_columns=["run_id", "source_id", "final_score", "priority_tier"],
            required_columns=("run_id", "source_id", "final_score"),
            table_name="lab.gaia_priority_results",
        )


def test_save_priority_results_skips_empty_frame_without_db_introspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой persist не должен доходить до relation-introspection и записи."""

    def fail_relation_columns(*args: Any, **kwargs: Any) -> list[str]:
        raise AssertionError("relation_columns should not be called for empty frame")

    monkeypatch.setattr(persist, "relation_columns", fail_relation_columns)

    persist.save_priority_results(
        df_priority=pd.DataFrame(),
        engine=cast(Engine, object()),
    )


def test_save_priority_results_writes_filtered_payload_into_target_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist должен писать только совместимый payload в нужную schema.table."""
    df_priority = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "source_id": 101,
                "ra": 10.0,
                "dec": -5.0,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "router_label": "K_dwarf",
                "quality_factor": 0.97,
                "reliability_factor": 0.97,
                "followup_factor": 0.92,
                "final_score": 0.72,
                "priority_tier": "HIGH",
                "reason_code": "HOST_SCORING",
                "router_model_version": "gaussian_router_v1",
                "host_model_version": "gaussian_host_field_v1",
                "debug_only": "drop-me",
            }
        ]
    )
    captured: dict[str, Any] = {}

    def fake_relation_columns(engine: Engine, relation_name: str) -> list[str]:
        assert relation_name == "lab.qa_priority_results"
        return [
            "run_id",
            "source_id",
            "ra",
            "dec",
            "predicted_spec_class",
            "predicted_evolution_stage",
            "router_label",
            "quality_factor",
            "reliability_factor",
            "followup_factor",
            "final_score",
            "priority_tier",
            "reason_code",
            "router_model_version",
            "host_model_version",
        ]

    def fake_to_sql(
        self: pd.DataFrame,
        *,
        name: str,
        schema: str | None = None,
        con: Engine,
        if_exists: str,
        index: bool,
        method: str,
        chunksize: int,
    ) -> None:
        captured["name"] = name
        captured["schema"] = schema
        captured["con"] = con
        captured["if_exists"] = if_exists
        captured["index"] = index
        captured["method"] = method
        captured["chunksize"] = chunksize
        captured["columns"] = self.columns.tolist()
        captured["row"] = self.iloc[0].to_dict()

    monkeypatch.setattr(persist, "relation_columns", fake_relation_columns)
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    engine = cast(Engine, object())
    persist.save_priority_results(
        df_priority=df_priority,
        engine=engine,
        table_name="lab.qa_priority_results",
    )

    assert captured == {
        "name": "qa_priority_results",
        "schema": "lab",
        "con": engine,
        "if_exists": "append",
        "index": False,
        "method": "multi",
        "chunksize": 1000,
        "columns": [
            "run_id",
            "source_id",
            "ra",
            "dec",
            "predicted_spec_class",
            "predicted_evolution_stage",
            "router_label",
            "quality_factor",
            "reliability_factor",
            "followup_factor",
            "final_score",
            "priority_tier",
            "reason_code",
            "router_model_version",
            "host_model_version",
        ],
        "row": {
            "run_id": "run_1",
            "source_id": 101,
            "ra": 10.0,
            "dec": -5.0,
            "predicted_spec_class": "K",
            "predicted_evolution_stage": "dwarf",
            "router_label": "K_dwarf",
            "quality_factor": 0.97,
            "reliability_factor": 0.97,
            "followup_factor": 0.92,
            "final_score": 0.72,
            "priority_tier": "HIGH",
            "reason_code": "HOST_SCORING",
            "router_model_version": "gaussian_router_v1",
            "host_model_version": "gaussian_host_field_v1",
        },
    }


def test_split_relation_name_uses_public_by_default() -> None:
    """Relation wrapper должен сохранять public-schema fallback и explicit schema."""
    assert split_relation_name("gaia_priority_results") == ("public", "gaia_priority_results")
    assert split_relation_name("lab.gaia_priority_results") == ("lab", "gaia_priority_results")
