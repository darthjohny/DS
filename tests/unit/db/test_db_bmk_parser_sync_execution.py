# Тестовый файл `test_db_bmk_parser_sync_execution.py` домена `db`.
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

from exohost.db.bmk_parser_sync_contracts import BmkParserSyncSummary
from exohost.db.bmk_parser_sync_execution import sync_bmk_parser_fields_downstream


class DummyCursor:
    # Минимальный cursor-stub для проверки orchestration без реальной БД.

    def __init__(
        self,
        fetch_results: list[tuple[object, ...] | None],
        *,
        rowcount: object = 0,
    ) -> None:
        self._fetch_results = fetch_results
        self.rowcount = rowcount
        self.executed_sql: list[str] = []
        self.closed = False

    def execute(self, operation: str) -> object:
        self.executed_sql.append(operation)
        return None

    def fetchone(self) -> tuple[object, ...] | None:
        if not self._fetch_results:
            return None
        return self._fetch_results.pop(0)

    def close(self) -> None:
        self.closed = True


class DummyConnection:
    # Минимальный connection-stub для проверки commit/rollback поведения.

    def __init__(self, cursor: DummyCursor) -> None:
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> DummyCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class DummyEngine:
    # Минимальный engine-stub для orchestration слоя.

    def __init__(self, connection: DummyConnection) -> None:
        self._connection = connection

    def raw_connection(self) -> DummyConnection:
        return self._connection


def _skip_validation(engine: Engine, *, source_relation_name: str) -> None:
    del engine, source_relation_name


def test_sync_bmk_parser_fields_downstream_returns_relation_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = DummyCursor([(2, 3, 4)], rowcount=7)
    connection = DummyConnection(cursor)
    engine = DummyEngine(connection)

    monkeypatch.setattr(
        "exohost.db.bmk_parser_sync_execution.validate_bmk_parser_sync_columns",
        _skip_validation,
    )
    monkeypatch.setattr(
        "exohost.db.bmk_parser_sync_execution.B_MK_PARSER_SYNC_TARGET_RELATION_NAMES",
        ("lab.test_target",),
    )

    summary = sync_bmk_parser_fields_downstream(cast(Engine, engine))

    assert isinstance(summary, BmkParserSyncSummary)
    assert summary.source_relation_name == "lab.gaia_mk_external_labeled"
    assert summary.relation_summaries == (
        summary.relation_summaries[0].__class__(
            relation_name="lab.test_target",
            rows_updated=7,
            ambiguous_ob_rows=2,
            ob_rows=3,
            o_rows=4,
        ),
    )
    assert connection.committed is True
    assert connection.rolled_back is False
    assert connection.closed is True
    assert cursor.closed is True


def test_sync_bmk_parser_fields_downstream_rolls_back_on_summary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = DummyCursor([None], rowcount=5)
    connection = DummyConnection(cursor)
    engine = DummyEngine(connection)

    monkeypatch.setattr(
        "exohost.db.bmk_parser_sync_execution.validate_bmk_parser_sync_columns",
        _skip_validation,
    )
    monkeypatch.setattr(
        "exohost.db.bmk_parser_sync_execution.B_MK_PARSER_SYNC_TARGET_RELATION_NAMES",
        ("lab.test_target",),
    )

    with pytest.raises(RuntimeError, match="summary query returned no rows"):
        sync_bmk_parser_fields_downstream(cast(Engine, engine))

    assert connection.committed is False
    assert connection.rolled_back is True
    assert connection.closed is True
    assert cursor.closed is True
