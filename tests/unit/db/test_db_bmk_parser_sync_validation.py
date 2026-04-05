# Тестовый файл `test_db_bmk_parser_sync_validation.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from typing import cast

import pytest
from sqlalchemy.engine import Engine

from exohost.db.bmk_parser_sync_contracts import (
    B_MK_PARSER_DERIVED_COLUMNS,
    B_MK_PARSER_SYNC_JOIN_COLUMNS,
    B_MK_PARSER_SYNC_TARGET_RELATION_NAMES,
)
from exohost.db.bmk_parser_sync_validation import validate_bmk_parser_sync_columns


def test_validate_bmk_parser_sync_columns_accepts_full_relation_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_columns = tuple(B_MK_PARSER_SYNC_JOIN_COLUMNS) + tuple(
        B_MK_PARSER_DERIVED_COLUMNS
    ) + ("external_catalog_name",)

    def fake_relation_columns(
        engine: object,
        relation_name: str,
        *,
        validate_identifiers: bool,
    ) -> tuple[str, ...]:
        assert validate_identifiers is True
        return required_columns

    monkeypatch.setattr(
        "exohost.db.bmk_parser_sync_validation.relation_columns",
        fake_relation_columns,
    )

    validate_bmk_parser_sync_columns(cast(Engine, object()))


def test_validate_bmk_parser_sync_columns_reports_missing_target_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_columns = tuple(B_MK_PARSER_SYNC_JOIN_COLUMNS) + tuple(
        B_MK_PARSER_DERIVED_COLUMNS
    ) + ("external_catalog_name",)
    broken_relation = B_MK_PARSER_SYNC_TARGET_RELATION_NAMES[0]

    def fake_relation_columns(
        engine: object,
        relation_name: str,
        *,
        validate_identifiers: bool,
    ) -> tuple[str, ...]:
        assert validate_identifiers is True
        if relation_name == broken_relation:
            return tuple(column for column in required_columns if column != "source_id")
        return required_columns

    monkeypatch.setattr(
        "exohost.db.bmk_parser_sync_validation.relation_columns",
        fake_relation_columns,
    )

    with pytest.raises(RuntimeError, match="missing required columns"):
        validate_bmk_parser_sync_columns(cast(Engine, object()))
