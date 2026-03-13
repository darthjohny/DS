"""Точечные тесты wrapper-ов `priority_pipeline.relations`."""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy.engine import Engine

import priority_pipeline.relations as pipeline_relations


def test_split_relation_name_wrapper_delegates_to_infra_helper(
    monkeypatch,
) -> None:
    """Wrapper должен напрямую делегировать разбор relation name в infra-layer."""
    captured: dict[str, Any] = {}

    def fake_split(relation_name: str) -> tuple[str, str]:
        captured["relation_name"] = relation_name
        return ("lab", "gaia_priority_results")

    monkeypatch.setattr(pipeline_relations, "_split_relation_name", fake_split)

    assert pipeline_relations.split_relation_name("lab.gaia_priority_results") == (
        "lab",
        "gaia_priority_results",
    )
    assert captured == {"relation_name": "lab.gaia_priority_results"}


def test_relation_exists_wrapper_forces_include_views(
    monkeypatch,
) -> None:
    """Wrapper production-layer должен всегда включать view introspection."""
    captured: dict[str, Any] = {}

    def fake_relation_exists(
        engine: Engine,
        relation_name: str,
        *,
        include_views: bool = False,
    ) -> bool:
        captured["engine"] = engine
        captured["relation_name"] = relation_name
        captured["include_views"] = include_views
        return True

    monkeypatch.setattr(pipeline_relations, "_relation_exists", fake_relation_exists)

    engine = cast(Engine, object())
    assert pipeline_relations.relation_exists(engine, "lab.gaia_priority_results") is True
    assert captured == {
        "engine": engine,
        "relation_name": "lab.gaia_priority_results",
        "include_views": True,
    }


def test_relation_columns_wrapper_returns_list_copy(
    monkeypatch,
) -> None:
    """Wrapper должен нормализовать infra tuple/iterable к обычному list[str]."""

    def fake_relation_columns(engine: Engine, relation_name: str) -> tuple[str, ...]:
        assert relation_name == "lab.gaia_priority_results"
        return ("run_id", "source_id", "final_score")

    monkeypatch.setattr(pipeline_relations, "_relation_columns", fake_relation_columns)

    columns = pipeline_relations.relation_columns(
        cast(Engine, object()),
        "lab.gaia_priority_results",
    )

    assert columns == ["run_id", "source_id", "final_score"]
