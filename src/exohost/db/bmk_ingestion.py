# Файл `bmk_ingestion.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from exohost.db.relations import (
    quote_identifier,
    quote_relation_name,
    validate_identifier,
)
from exohost.ingestion.bmk.contracts import (
    B_MK_FILTERED_COLUMNS,
    B_MK_RAW_COLUMNS,
    B_MK_REJECTED_COLUMNS,
    BmkExportPaths,
)

B_MK_SCHEMA_NAME = "lab"
B_MK_RAW_RELATION_NAME = "lab.gaia_mk_external_raw"
B_MK_FILTERED_RELATION_NAME = "lab.gaia_mk_external_filtered"
B_MK_REJECTED_RELATION_NAME = "lab.gaia_mk_external_rejected"


@dataclass(frozen=True, slots=True)
class BmkDatabaseLoadSummary:
    # Фактические row counts после полной загрузки трех B/mk relation.
    raw_relation_name: str
    filtered_relation_name: str
    rejected_relation_name: str
    raw_rows_loaded: int
    filtered_rows_loaded: int
    rejected_rows_loaded: int


def build_bmk_schema_sql() -> tuple[str, ...]:
    # Собираем DDL первых трех relation B/mk-ветки.
    schema_sql = f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(B_MK_SCHEMA_NAME)}"
    raw_sql = f"""
CREATE TABLE IF NOT EXISTS {quote_relation_name(B_MK_RAW_RELATION_NAME, validate_identifiers=True)} (
    external_row_id BIGINT PRIMARY KEY,
    external_catalog_name TEXT NOT NULL,
    external_object_id TEXT,
    ra_deg DOUBLE PRECISION NOT NULL,
    dec_deg DOUBLE PRECISION NOT NULL,
    raw_sptype TEXT NOT NULL,
    raw_magnitude DOUBLE PRECISION,
    raw_source_bibcode TEXT,
    raw_notes TEXT,
    imported_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""".strip()
    filtered_sql = f"""
CREATE TABLE IF NOT EXISTS {quote_relation_name(B_MK_FILTERED_RELATION_NAME, validate_identifiers=True)} (
    external_row_id BIGINT PRIMARY KEY,
    external_catalog_name TEXT NOT NULL,
    external_object_id TEXT,
    ra_deg DOUBLE PRECISION NOT NULL,
    dec_deg DOUBLE PRECISION NOT NULL,
    raw_sptype TEXT NOT NULL,
    raw_magnitude DOUBLE PRECISION,
    raw_source_bibcode TEXT,
    raw_notes TEXT,
    spectral_prefix TEXT NOT NULL,
    spectral_class TEXT NOT NULL,
    spectral_subclass INTEGER,
    luminosity_class TEXT,
    parse_status TEXT NOT NULL,
    parse_note TEXT,
    has_supported_prefix BOOLEAN NOT NULL,
    has_coordinates BOOLEAN NOT NULL,
    has_raw_sptype BOOLEAN NOT NULL,
    ready_for_gaia_crossmatch BOOLEAN NOT NULL,
    filtered_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""".strip()
    rejected_sql = f"""
CREATE TABLE IF NOT EXISTS {quote_relation_name(B_MK_REJECTED_RELATION_NAME, validate_identifiers=True)} (
    external_row_id BIGINT PRIMARY KEY,
    external_catalog_name TEXT,
    external_object_id TEXT,
    ra_deg DOUBLE PRECISION,
    dec_deg DOUBLE PRECISION,
    raw_sptype TEXT,
    raw_magnitude DOUBLE PRECISION,
    raw_source_bibcode TEXT,
    raw_notes TEXT,
    spectral_prefix TEXT,
    reject_reason TEXT NOT NULL,
    rejected_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""".strip()
    return schema_sql, raw_sql, filtered_sql, rejected_sql


def build_copy_from_stdin_sql(
    relation_name: str,
    *,
    columns: tuple[str, ...],
) -> str:
    # Строим безопасный COPY FROM STDIN для конкретной relation.
    relation_sql = quote_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    columns_sql = ", ".join(quote_identifier(validate_identifier(column_name)) for column_name in columns)
    return (
        f"COPY {relation_sql} ({columns_sql}) "
        "FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')"
    )


def load_bmk_exports_into_db(
    engine: Engine,
    export_paths: BmkExportPaths,
) -> BmkDatabaseLoadSummary:
    # Создаем relation, полностью перезаливаем raw/filtered/rejected и считаем row counts.
    dbapi_connection = engine.raw_connection()
    cursor = dbapi_connection.cursor()
    try:
        for statement in build_bmk_schema_sql():
            cursor.execute(statement)

        _replace_relation_from_csv(
            cursor,
            relation_name=B_MK_RAW_RELATION_NAME,
            csv_path=export_paths.raw_csv_path,
            columns=B_MK_RAW_COLUMNS,
        )
        _replace_relation_from_csv(
            cursor,
            relation_name=B_MK_FILTERED_RELATION_NAME,
            csv_path=export_paths.filtered_csv_path,
            columns=B_MK_FILTERED_COLUMNS,
        )
        _replace_relation_from_csv(
            cursor,
            relation_name=B_MK_REJECTED_RELATION_NAME,
            csv_path=export_paths.rejected_csv_path,
            columns=B_MK_REJECTED_COLUMNS,
        )

        load_summary = BmkDatabaseLoadSummary(
            raw_relation_name=B_MK_RAW_RELATION_NAME,
            filtered_relation_name=B_MK_FILTERED_RELATION_NAME,
            rejected_relation_name=B_MK_REJECTED_RELATION_NAME,
            raw_rows_loaded=_count_relation_rows(cursor, B_MK_RAW_RELATION_NAME),
            filtered_rows_loaded=_count_relation_rows(cursor, B_MK_FILTERED_RELATION_NAME),
            rejected_rows_loaded=_count_relation_rows(cursor, B_MK_REJECTED_RELATION_NAME),
        )
        dbapi_connection.commit()
        return load_summary
    except Exception:
        dbapi_connection.rollback()
        raise
    finally:
        cursor.close()
        dbapi_connection.close()


def _replace_relation_from_csv(
    cursor: Any,
    *,
    relation_name: str,
    csv_path: Path,
    columns: tuple[str, ...],
) -> None:
    # Полностью обновляем relation из подготовленного staging CSV.
    relation_sql = quote_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    copy_sql = build_copy_from_stdin_sql(
        relation_name,
        columns=columns,
    )
    cursor.execute(f"TRUNCATE TABLE {relation_sql}")
    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        cursor.copy_expert(copy_sql, input_file)


def _count_relation_rows(
    cursor: Any,
    relation_name: str,
) -> int:
    # Считаем строки relation сразу после загрузки в том же transaction scope.
    relation_sql = quote_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    cursor.execute(f"SELECT COUNT(*) FROM {relation_sql}")
    result = cursor.fetchone()
    if result is None:
        raise RuntimeError(f"COUNT(*) returned no rows for relation: {relation_name}")
    return int(result[0])
