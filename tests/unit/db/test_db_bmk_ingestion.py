# Тестовый файл `test_db_bmk_ingestion.py` домена `db`.
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
from io import TextIOBase
from pathlib import Path
from typing import cast

from sqlalchemy.engine import Engine

from exohost.db.bmk_ingestion import (
    B_MK_FILTERED_RELATION_NAME,
    B_MK_RAW_RELATION_NAME,
    B_MK_REJECTED_RELATION_NAME,
    BmkDatabaseLoadSummary,
    build_bmk_schema_sql,
    build_copy_from_stdin_sql,
    load_bmk_exports_into_db,
)
from exohost.ingestion.bmk import BmkExportPaths


def test_build_bmk_schema_sql_contains_three_relation_definitions() -> None:
    schema_sql = build_bmk_schema_sql()

    assert len(schema_sql) == 4
    assert "CREATE SCHEMA IF NOT EXISTS" in schema_sql[0]
    assert B_MK_RAW_RELATION_NAME.split(".", 1)[1] in schema_sql[1]
    assert B_MK_FILTERED_RELATION_NAME.split(".", 1)[1] in schema_sql[2]
    assert B_MK_REJECTED_RELATION_NAME.split(".", 1)[1] in schema_sql[3]


def test_build_copy_from_stdin_sql_uses_csv_header_and_null_empty_string() -> None:
    copy_sql = build_copy_from_stdin_sql(
        "lab.gaia_mk_external_raw",
        columns=("external_row_id", "raw_sptype"),
    )

    assert 'COPY "lab"."gaia_mk_external_raw"' in copy_sql
    assert '"external_row_id", "raw_sptype"' in copy_sql
    assert "FORMAT CSV" in copy_sql
    assert "HEADER TRUE" in copy_sql
    assert "NULL ''" in copy_sql


@dataclass
class FakeCursor:
    # Минимальный DBAPI cursor для проверки COPY/DDL без реальной БД.
    counts: list[tuple[int]] = field(default_factory=lambda: [(2,), (1,), (3,)])
    executed: list[str] = field(default_factory=list)
    copied: list[tuple[str, str]] = field(default_factory=list)
    closed: bool = False

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def copy_expert(self, sql: str, file_obj: TextIOBase) -> None:
        self.copied.append((sql, file_obj.read()))

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


def write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_load_bmk_exports_into_db_truncates_copies_and_counts(tmp_path: Path) -> None:
    export_paths = BmkExportPaths(
        raw_csv_path=tmp_path / "raw.csv",
        filtered_csv_path=tmp_path / "filtered.csv",
        rejected_csv_path=tmp_path / "rejected.csv",
    )
    write_csv(
        export_paths.raw_csv_path,
        "external_row_id,external_catalog_name,external_object_id,ra_deg,dec_deg,raw_sptype,raw_magnitude,raw_source_bibcode,raw_notes",
        ["0,bmk,Star A,10.5,-5.25,G2V,,2014yCat....1.2023S,"],
    )
    write_csv(
        export_paths.filtered_csv_path,
        "external_row_id,external_catalog_name,external_object_id,ra_deg,dec_deg,raw_sptype,raw_magnitude,raw_source_bibcode,raw_notes,spectral_prefix,spectral_class,spectral_subclass,luminosity_class,parse_status,parse_note,has_supported_prefix,has_coordinates,has_raw_sptype,ready_for_gaia_crossmatch",
        ["0,bmk,Star A,10.5,-5.25,G2V,,2014yCat....1.2023S,,G,G,2,V,parsed,,True,True,True,True"],
    )
    write_csv(
        export_paths.rejected_csv_path,
        "external_row_id,external_catalog_name,external_object_id,ra_deg,dec_deg,raw_sptype,raw_magnitude,raw_source_bibcode,raw_notes,spectral_prefix,reject_reason",
        ["3,bmk,Star D,,,G2V,12.2,2012ApJS..203...21D,,,missing_coordinates"],
    )
    engine = FakeEngine()

    load_summary = load_bmk_exports_into_db(cast(Engine, engine), export_paths)

    assert load_summary == BmkDatabaseLoadSummary(
        raw_relation_name=B_MK_RAW_RELATION_NAME,
        filtered_relation_name=B_MK_FILTERED_RELATION_NAME,
        rejected_relation_name=B_MK_REJECTED_RELATION_NAME,
        raw_rows_loaded=2,
        filtered_rows_loaded=1,
        rejected_rows_loaded=3,
    )
    assert engine.connection.committed is True
    assert engine.connection.rolled_back is False
    assert engine.connection.closed is True
    assert engine.connection.cursor_obj.closed is True
    assert len(engine.connection.cursor_obj.copied) == 3
    assert any("TRUNCATE TABLE" in sql for sql in engine.connection.cursor_obj.executed)
    assert any("SELECT COUNT(*)" in sql for sql in engine.connection.cursor_obj.executed)
