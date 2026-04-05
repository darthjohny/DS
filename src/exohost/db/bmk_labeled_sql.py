# Файл `bmk_labeled_sql.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.db.bmk_crossmatch import validate_xmatch_batch_id
from exohost.db.bmk_labeled_contracts import (
    B_MK_EXTERNAL_LABELED_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
)
from exohost.db.relations import (
    quote_identifier,
    quote_relation_name,
    split_relation_name,
    validate_identifier,
)


def build_bmk_external_labeled_schema_sql(
    target_relation_name: str = B_MK_EXTERNAL_LABELED_RELATION_NAME,
) -> tuple[str, ...]:
    # Собираем DDL для canonical normalized labeled relation.
    schema_name, table_name = split_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    relation_sql = quote_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    source_index_name = validate_identifier(f"{table_name}_source_id_idx")
    schema_sql = f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema_name)}"
    table_sql = f"""
CREATE TABLE IF NOT EXISTS {relation_sql} (
    xmatch_batch_id TEXT NOT NULL,
    source_id BIGINT NOT NULL,
    external_row_id BIGINT NOT NULL,
    external_catalog_name TEXT NOT NULL,
    external_object_id TEXT,
    raw_sptype TEXT NOT NULL,
    spectral_class TEXT NOT NULL,
    spectral_subclass INTEGER,
    luminosity_class TEXT,
    peculiarity_suffix TEXT,
    label_parse_status TEXT NOT NULL,
    label_parse_notes TEXT,
    xmatch_separation_arcsec DOUBLE PRECISION NOT NULL,
    has_source_conflict BOOLEAN NOT NULL,
    source_conflict_count INTEGER NOT NULL,
    labeled_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (xmatch_batch_id, external_row_id, source_id)
)
""".strip()
    source_index_sql = f"""
CREATE INDEX IF NOT EXISTS {quote_identifier(source_index_name)}
ON {relation_sql} ("source_id")
""".strip()
    return schema_sql, table_sql, source_index_sql


def build_delete_bmk_external_labeled_batch_sql(
    target_relation_name: str = B_MK_EXTERNAL_LABELED_RELATION_NAME,
    *,
    xmatch_batch_id: str,
) -> str:
    # Удаляем только один labeled batch, не трогая остальные materialization запуски.
    batch_id_sql = quote_text_literal(validate_xmatch_batch_id(xmatch_batch_id))
    relation_sql = quote_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    return (
        f'DELETE FROM {relation_sql} '
        f'WHERE "xmatch_batch_id" = {batch_id_sql}'
    )


def build_bmk_external_labeled_source_query(
    *,
    filtered_relation_name: str = B_MK_EXTERNAL_LABELED_SOURCE_FILTERED_RELATION_NAME,
    crossmatch_relation_name: str = B_MK_EXTERNAL_LABELED_SOURCE_CROSSMATCH_RELATION_NAME,
    xmatch_batch_id: str,
    limit: int | None = None,
) -> str:
    # Строим детерминированный selected join для локального labeled materialization.
    batch_id_sql = quote_text_literal(validate_xmatch_batch_id(xmatch_batch_id))
    filtered_relation_sql = quote_relation_name(
        filtered_relation_name,
        validate_identifiers=True,
    )
    crossmatch_relation_sql = quote_relation_name(
        crossmatch_relation_name,
        validate_identifiers=True,
    )
    limit_sql = f"\nLIMIT {limit}" if limit is not None else ""
    return f"""
SELECT
    c."xmatch_batch_id",
    c."source_id",
    c."external_row_id",
    c."xmatch_separation_arcsec",
    f."external_catalog_name",
    f."external_object_id",
    f."raw_sptype",
    COUNT(*) OVER (
        PARTITION BY c."source_id"
    ) AS "source_conflict_count"
FROM {crossmatch_relation_sql} AS c
INNER JOIN {filtered_relation_sql} AS f
    ON f."external_row_id" = c."external_row_id"
WHERE c."xmatch_batch_id" = {batch_id_sql}
  AND c."xmatch_selected" IS TRUE
ORDER BY c."external_row_id" ASC
{limit_sql}
""".strip()


def quote_text_literal(value: str) -> str:
    # Экранируем текстовый literal для узкого внутреннего SQL-контура.
    return "'" + value.replace("'", "''") + "'"
