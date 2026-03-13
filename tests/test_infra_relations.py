"""Точечные тесты introspection-helper-ов relation."""

from __future__ import annotations

from typing import Any, cast

import pytest
from sqlalchemy.engine import Engine

import infra.relations as infra_relations


def test_split_relation_name_supports_default_public_and_explicit_schema() -> None:
    """Helper должен поддерживать и implicit public, и explicit schema."""
    assert infra_relations.split_relation_name("gaia_priority_results") == (
        "public",
        "gaia_priority_results",
    )
    assert infra_relations.split_relation_name("lab.gaia_priority_results") == (
        "lab",
        "gaia_priority_results",
    )


def test_split_relation_name_validates_identifiers_when_requested() -> None:
    """При включённой проверке helper должен отвергать небезопасные идентификаторы."""
    with pytest.raises(ValueError, match="Invalid schema name"):
        infra_relations.split_relation_name(
            "bad-schema.gaia_priority_results",
            validate_identifiers=True,
        )

    with pytest.raises(ValueError, match="Invalid table name"):
        infra_relations.split_relation_name(
            "lab.bad-table",
            validate_identifiers=True,
        )


def test_relation_exists_checks_tables_and_views_via_inspector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relation helper должен учитывать table/view policy через inspector."""
    captured: dict[str, Any] = {}

    class FakeInspector:
        def get_table_names(self, schema: str) -> list[str]:
            captured["table_schema"] = schema
            return ["gaia_priority_results"]

        def get_view_names(self, schema: str) -> list[str]:
            captured["view_schema"] = schema
            return ["gaia_priority_view"]

    monkeypatch.setattr(infra_relations, "sa_inspect", lambda engine: FakeInspector())

    engine = cast(Engine, object())

    assert infra_relations.relation_exists(engine, "lab.gaia_priority_results") is True
    assert infra_relations.relation_exists(engine, "lab.gaia_priority_view") is True
    assert (
        infra_relations.relation_exists(
            engine,
            "lab.gaia_priority_view",
            include_views=False,
        )
        is False
    )
    assert captured == {
        "table_schema": "lab",
        "view_schema": "lab",
    }


def test_relation_exists_passes_validation_flag_into_name_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation flag должен доходить до split helper при relation introspection."""
    captured: dict[str, Any] = {}

    def fake_split_relation_name(
        relation_name: str,
        *,
        validate_identifiers: bool = False,
    ) -> tuple[str, str]:
        captured["relation_name"] = relation_name
        captured["validate_identifiers"] = validate_identifiers
        return ("lab", "gaia_priority_results")

    class FakeInspector:
        def get_table_names(self, schema: str) -> list[str]:
            return ["gaia_priority_results"]

        def get_view_names(self, schema: str) -> list[str]:
            return []

    monkeypatch.setattr(infra_relations, "split_relation_name", fake_split_relation_name)
    monkeypatch.setattr(infra_relations, "sa_inspect", lambda engine: FakeInspector())

    assert (
        infra_relations.relation_exists(
            cast(Engine, object()),
            "lab.gaia_priority_results",
            validate_identifiers=True,
        )
        is True
    )
    assert captured == {
        "relation_name": "lab.gaia_priority_results",
        "validate_identifiers": True,
    }


def test_relation_columns_returns_ordered_column_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Column helper должен возвращать упорядоченный tuple имён колонок."""

    class FakeInspector:
        def get_columns(self, table_name: str, schema: str) -> list[dict[str, object]]:
            assert table_name == "gaia_priority_results"
            assert schema == "lab"
            return [
                {"name": "run_id"},
                {"name": "source_id"},
                {"name": "final_score"},
            ]

    monkeypatch.setattr(infra_relations, "sa_inspect", lambda engine: FakeInspector())

    columns = infra_relations.relation_columns(
        cast(Engine, object()),
        "lab.gaia_priority_results",
    )

    assert columns == ("run_id", "source_id", "final_score")
