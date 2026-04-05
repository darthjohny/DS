# Файл `bmk_crossmatch.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.engine import Engine

from exohost.db.relations import (
    quote_identifier,
    quote_relation_name,
    relation_columns,
    split_relation_name,
    validate_identifier,
)

B_MK_GAIA_XMATCH_RAW_SOURCE_RELATION_NAME = "public.raw_landing_table"
B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME = "lab.gaia_mk_external_crossmatch"
B_MK_EXTERNAL_CROSSMATCH_REQUIRED_SOURCE_COLUMNS = (
    "external_row_id",
    "source_id",
    "SEPARATION",
)
XMATCH_BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_:-]+$")


@dataclass(frozen=True, slots=True)
class BmkCrossmatchMaterializationSummary:
    # Фактические row counts после materialization batch-а в canonical crossmatch relation.
    source_relation_name: str
    target_relation_name: str
    xmatch_batch_id: str
    rows_loaded: int
    distinct_external_rows: int
    selected_rows: int
    multi_match_external_rows: int


def build_bmk_crossmatch_schema_sql(
    target_relation_name: str = B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME,
) -> tuple[str, ...]:
    # Собираем DDL для canonical post-Gaia crossmatch relation.
    schema_name, table_name = split_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    relation_sql = quote_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    source_index_name = validate_identifier(f"{table_name}_source_id_idx")
    selected_index_name = validate_identifier(f"{table_name}_batch_selected_idx")
    schema_sql = f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema_name)}"
    table_sql = f"""
CREATE TABLE IF NOT EXISTS {relation_sql} (
    xmatch_batch_id TEXT NOT NULL,
    external_row_id BIGINT NOT NULL,
    source_id BIGINT NOT NULL,
    xmatch_separation_arcsec DOUBLE PRECISION NOT NULL,
    xmatch_rank INTEGER NOT NULL,
    xmatch_selected BOOLEAN NOT NULL,
    matched_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (xmatch_batch_id, external_row_id, source_id)
)
""".strip()
    source_index_sql = f"""
CREATE INDEX IF NOT EXISTS {quote_identifier(source_index_name)}
ON {relation_sql} ("source_id")
""".strip()
    selected_index_sql = f"""
CREATE INDEX IF NOT EXISTS {quote_identifier(selected_index_name)}
ON {relation_sql} ("xmatch_batch_id", "external_row_id", "xmatch_selected", "xmatch_rank")
""".strip()
    return schema_sql, table_sql, source_index_sql, selected_index_sql


def build_delete_bmk_crossmatch_batch_sql(
    target_relation_name: str = B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME,
    *,
    xmatch_batch_id: str,
) -> str:
    # Удаляем только один batch, чтобы не трогать другие materialized куски.
    batch_id_sql = _quote_text_literal(validate_xmatch_batch_id(xmatch_batch_id))
    relation_sql = quote_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    return (
        f'DELETE FROM {relation_sql} '
        f'WHERE "xmatch_batch_id" = {batch_id_sql}'
    )


def build_bmk_crossmatch_insert_sql(
    source_relation_name: str = B_MK_GAIA_XMATCH_RAW_SOURCE_RELATION_NAME,
    *,
    target_relation_name: str = B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME,
    xmatch_batch_id: str,
) -> str:
    # Строим deterministic insert с rank/select логикой поверх raw Gaia export.
    batch_id_sql = _quote_text_literal(validate_xmatch_batch_id(xmatch_batch_id))
    source_relation_sql = quote_relation_name(
        source_relation_name,
        validate_identifiers=True,
    )
    target_relation_sql = quote_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    return f"""
WITH ranked_matches AS (
    SELECT
        "external_row_id"::BIGINT AS external_row_id,
        "source_id"::BIGINT AS source_id,
        "SEPARATION"::DOUBLE PRECISION * 3600.0 AS xmatch_separation_arcsec,
        ROW_NUMBER() OVER (
            PARTITION BY "external_row_id"
            ORDER BY
                "SEPARATION"::DOUBLE PRECISION ASC,
                "source_id"::BIGINT ASC
        ) AS xmatch_rank
    FROM {source_relation_sql}
    WHERE "external_row_id" IS NOT NULL
      AND "source_id" IS NOT NULL
      AND "SEPARATION" IS NOT NULL
)
INSERT INTO {target_relation_sql} (
    "xmatch_batch_id",
    "external_row_id",
    "source_id",
    "xmatch_separation_arcsec",
    "xmatch_rank",
    "xmatch_selected"
)
SELECT
    {batch_id_sql},
    external_row_id,
    source_id,
    xmatch_separation_arcsec,
    xmatch_rank,
    xmatch_rank = 1
FROM ranked_matches
ORDER BY external_row_id ASC, xmatch_rank ASC
""".strip()


def materialize_bmk_crossmatch_relation(
    engine: Engine,
    *,
    source_relation_name: str = B_MK_GAIA_XMATCH_RAW_SOURCE_RELATION_NAME,
    target_relation_name: str = B_MK_EXTERNAL_CROSSMATCH_RELATION_NAME,
    xmatch_batch_id: str,
) -> BmkCrossmatchMaterializationSummary:
    # Строим canonical crossmatch relation из wide raw landing relation.
    validate_xmatch_batch_id(xmatch_batch_id)
    _validate_required_source_columns(
        engine,
        source_relation_name,
    )
    dbapi_connection = engine.raw_connection()
    cursor = dbapi_connection.cursor()
    try:
        for statement in build_bmk_crossmatch_schema_sql(target_relation_name):
            cursor.execute(statement)

        cursor.execute(
            build_delete_bmk_crossmatch_batch_sql(
                target_relation_name,
                xmatch_batch_id=xmatch_batch_id,
            )
        )
        cursor.execute(
            build_bmk_crossmatch_insert_sql(
                source_relation_name,
                target_relation_name=target_relation_name,
                xmatch_batch_id=xmatch_batch_id,
            )
        )

        load_summary = BmkCrossmatchMaterializationSummary(
            source_relation_name=source_relation_name,
            target_relation_name=target_relation_name,
            xmatch_batch_id=xmatch_batch_id,
            rows_loaded=_count_batch_rows(
                cursor,
                relation_name=target_relation_name,
                xmatch_batch_id=xmatch_batch_id,
            ),
            distinct_external_rows=_count_batch_distinct_external_rows(
                cursor,
                relation_name=target_relation_name,
                xmatch_batch_id=xmatch_batch_id,
            ),
            selected_rows=_count_batch_selected_rows(
                cursor,
                relation_name=target_relation_name,
                xmatch_batch_id=xmatch_batch_id,
            ),
            multi_match_external_rows=_count_batch_multi_match_external_rows(
                cursor,
                relation_name=target_relation_name,
                xmatch_batch_id=xmatch_batch_id,
            ),
        )
        dbapi_connection.commit()
        return load_summary
    except Exception:
        dbapi_connection.rollback()
        raise
    finally:
        cursor.close()
        dbapi_connection.close()


def validate_xmatch_batch_id(xmatch_batch_id: str) -> str:
    # Batch id оставляем коротким и SQL-safe для reproducible materialization.
    if not XMATCH_BATCH_ID_PATTERN.fullmatch(xmatch_batch_id):
        raise ValueError(f"Invalid xmatch batch id: {xmatch_batch_id}")
    return xmatch_batch_id


def _validate_required_source_columns(
    engine: Engine,
    source_relation_name: str,
) -> None:
    # Проверяем, что wide raw relation содержит минимальный набор crossmatch полей.
    source_columns = set(relation_columns(engine, source_relation_name))
    missing_columns = [
        column_name
        for column_name in B_MK_EXTERNAL_CROSSMATCH_REQUIRED_SOURCE_COLUMNS
        if column_name not in source_columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required source columns for B/mk crossmatch materialization: "
            + ", ".join(missing_columns)
        )


def _count_batch_rows(
    cursor: Any,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем все строки materialized batch-а.
    return _fetch_single_count(
        cursor,
        f"""
SELECT COUNT(*)
FROM {quote_relation_name(relation_name, validate_identifiers=True)}
WHERE "xmatch_batch_id" = {_quote_text_literal(xmatch_batch_id)}
""".strip(),
    )


def _count_batch_distinct_external_rows(
    cursor: Any,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем число уникальных external rows внутри materialized batch-а.
    return _fetch_single_count(
        cursor,
        f"""
SELECT COUNT(DISTINCT "external_row_id")
FROM {quote_relation_name(relation_name, validate_identifiers=True)}
WHERE "xmatch_batch_id" = {_quote_text_literal(xmatch_batch_id)}
""".strip(),
    )


def _count_batch_selected_rows(
    cursor: Any,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем число выбранных рабочих match-ей в batch-е.
    return _fetch_single_count(
        cursor,
        f"""
SELECT COUNT(*)
FROM {quote_relation_name(relation_name, validate_identifiers=True)}
WHERE "xmatch_batch_id" = {_quote_text_literal(xmatch_batch_id)}
  AND "xmatch_selected" IS TRUE
""".strip(),
    )


def _count_batch_multi_match_external_rows(
    cursor: Any,
    *,
    relation_name: str,
    xmatch_batch_id: str,
) -> int:
    # Считаем, сколько external rows дали больше одного Gaia-кандидата.
    return _fetch_single_count(
        cursor,
        f"""
SELECT COUNT(*)
FROM (
    SELECT "external_row_id"
    FROM {quote_relation_name(relation_name, validate_identifiers=True)}
    WHERE "xmatch_batch_id" = {_quote_text_literal(xmatch_batch_id)}
    GROUP BY "external_row_id"
    HAVING COUNT(*) > 1
) AS multi_match_rows
""".strip(),
    )


def _fetch_single_count(
    cursor: Any,
    sql: str,
) -> int:
    # Выполняем COUNT-запрос и возвращаем целое значение.
    cursor.execute(sql)
    result = cursor.fetchone()
    if result is None:
        raise RuntimeError("COUNT query returned no rows")
    return int(result[0])


def _quote_text_literal(value: str) -> str:
    # Экранируем текстовый literal для узкого внутреннего SQL-контура.
    return "'" + value.replace("'", "''") + "'"
