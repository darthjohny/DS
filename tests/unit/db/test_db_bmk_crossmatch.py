# Тестовый файл `test_db_bmk_crossmatch.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from sqlalchemy.engine import Engine

from exohost.db.bmk_crossmatch import (
    B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME,
    B_MK_EXTERNAL_CROSSMATCH_REQUIRED_SOURCE_COLUMNS,
    B_MK_GAIA_XMATCH_RAW_SOURCE_RELATION_NAME,
    BmkCrossmatchMaterializationSummary,
    build_bmk_crossmatch_insert_sql,
    build_bmk_crossmatch_schema_sql,
    build_delete_bmk_crossmatch_batch_sql,
    materialize_bmk_crossmatch_relation,
    validate_xmatch_batch_id,
)


def test_build_bmk_crossmatch_schema_sql_contains_relation_and_indexes() -> None:
    schema_sql = build_bmk_crossmatch_schema_sql()

    assert len(schema_sql) == 4
    assert "CREATE SCHEMA IF NOT EXISTS" in schema_sql[0]
    assert B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME.split(".", 1)[1] in schema_sql[1]
    assert "PRIMARY KEY" in schema_sql[1]
    assert "xmatch_batch_id" in schema_sql[1]
    assert "CREATE INDEX IF NOT EXISTS" in schema_sql[2]
    assert "source_id" in schema_sql[2]
    assert "xmatch_selected" in schema_sql[3]


def test_build_delete_bmk_crossmatch_batch_sql_targets_one_batch() -> None:
    sql = build_delete_bmk_crossmatch_batch_sql(
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
    )

    assert 'DELETE FROM "lab"."gaia_mk_external_crossmatch"' in sql
    assert '"xmatch_batch_id" =' in sql
    assert "xmatch_bmk_gaia_dr3__2026_03_26" in sql


def test_build_bmk_crossmatch_insert_sql_uses_rank_and_selected_logic() -> None:
    sql = build_bmk_crossmatch_insert_sql(
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
    )

    assert 'FROM "public"."raw_landing_table"' in sql
    assert 'INSERT INTO "lab"."gaia_mk_external_crossmatch"' in sql
    assert "ROW_NUMBER() OVER" in sql
    assert 'PARTITION BY "external_row_id"' in sql
    assert '"SEPARATION"::DOUBLE PRECISION * 3600.0 AS xmatch_separation_arcsec' in sql
    assert '"SEPARATION"::DOUBLE PRECISION ASC' in sql
    assert 'xmatch_rank = 1' in sql


def test_validate_xmatch_batch_id_rejects_invalid_value() -> None:
    try:
        validate_xmatch_batch_id("bad batch id")
    except ValueError as exc:
        assert "Invalid xmatch batch id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid batch id")


@dataclass
class FakeCursor:
    # Минимальный DBAPI cursor для проверки DDL/insert/count без реальной БД.
    counts: list[tuple[int]] = field(
        default_factory=lambda: [(824038,), (824038,), (824038,), (13905,)]
    )
    executed: list[str] = field(default_factory=list)
    closed: bool = False

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def fetchone(self) -> tuple[int]:
        return self.counts.pop(0)

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeConnection:
    # Минимальный raw DBAPI connection для проверки transaction-границы.
    cursor_obj: FakeCursor = field(default_factory=FakeCursor)
    committed: bool = False
    rolled_back: bool = False
    closed: bool = False

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeEngine:
    # Минимальный SQLAlchemy-like engine с raw_connection().
    connection: FakeConnection = field(default_factory=FakeConnection)

    def raw_connection(self) -> FakeConnection:
        return self.connection


def test_materialize_bmk_crossmatch_relation_creates_and_counts_batch(
    monkeypatch,
) -> None:
    engine = FakeEngine()

    monkeypatch.setattr(
        "exohost.db.bmk_crossmatch.relation_columns",
        lambda _engine, _relation_name: B_MK_EXTERNAL_CROSSMATCH_REQUIRED_SOURCE_COLUMNS,
    )

    load_summary = materialize_bmk_crossmatch_relation(
        cast(Engine, engine),
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
    )

    assert load_summary == BmkCrossmatchMaterializationSummary(
        source_relation_name=B_MK_GAIA_XMATCH_RAW_SOURCE_RELATION_NAME,
        target_relation_name=B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME,
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        rows_loaded=824038,
        distinct_external_rows=824038,
        selected_rows=824038,
        multi_match_external_rows=13905,
    )
    assert engine.connection.committed is True
    assert engine.connection.rolled_back is False
    assert engine.connection.closed is True
    assert engine.connection.cursor_obj.closed is True
    assert any("CREATE TABLE IF NOT EXISTS" in sql for sql in engine.connection.cursor_obj.executed)
    assert any("DELETE FROM" in sql for sql in engine.connection.cursor_obj.executed)
    assert any("ROW_NUMBER() OVER" in sql for sql in engine.connection.cursor_obj.executed)
    assert any("COUNT(DISTINCT \"external_row_id\")" in sql for sql in engine.connection.cursor_obj.executed)


def test_materialize_bmk_crossmatch_relation_requires_source_columns(
    monkeypatch,
) -> None:
    engine = FakeEngine()

    monkeypatch.setattr(
        "exohost.db.bmk_crossmatch.relation_columns",
        lambda _engine, _relation_name: ("external_row_id", "source_id"),
    )

    try:
        materialize_bmk_crossmatch_relation(
            cast(Engine, engine),
            xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
        )
    except ValueError as exc:
        assert "Missing required source columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing source columns")
