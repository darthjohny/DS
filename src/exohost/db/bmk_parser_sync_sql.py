# Файл `bmk_parser_sync_sql.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.db.bmk_parser_sync_contracts import (
    B_MK_PARSER_DERIVED_COLUMNS,
    B_MK_PARSER_SYNC_JOIN_COLUMNS,
    B_MK_PARSER_SYNC_SOURCE_RELATION_NAME,
    B_MK_QUALITY_GATED_RELATION_NAME,
    B_MK_QUALITY_SUMMARY_RELATION_NAME,
    B_MK_TRAINING_REFERENCE_RELATION_NAME,
    B_MK_TRAINING_SUMMARY_RELATION_NAME,
    B_MK_UNKNOWN_REVIEW_RELATION_NAME,
    B_MK_UNKNOWN_SUMMARY_RELATION_NAME,
)
from exohost.db.relations import quote_relation_name


def build_bmk_parser_sync_update_sql(
    target_relation_name: str,
    *,
    source_relation_name: str = B_MK_PARSER_SYNC_SOURCE_RELATION_NAME,
) -> str:
    # Собираем точечный UPDATE parser-derived полей для одной downstream relation.
    source_relation_sql = quote_relation_name(
        source_relation_name,
        validate_identifiers=True,
    )
    target_relation_sql = quote_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    assignments_sql = ",\n    ".join(
        f'"{column_name}" = source_relation."{column_name}"'
        for column_name in B_MK_PARSER_DERIVED_COLUMNS
    )
    distinct_checks_sql = " OR\n        ".join(
        "target_relation."
        f'"{column_name}" IS DISTINCT FROM source_relation."{column_name}"'
        for column_name in B_MK_PARSER_DERIVED_COLUMNS
    )
    join_checks_sql = "\n   AND ".join(
        "target_relation."
        f'"{column_name}" = source_relation."{column_name}"'
        for column_name in B_MK_PARSER_SYNC_JOIN_COLUMNS
    )
    return f"""
UPDATE {target_relation_sql} AS target_relation
SET
    {assignments_sql}
FROM {source_relation_sql} AS source_relation
WHERE {join_checks_sql}
  AND target_relation."external_catalog_name" = 'bmk'
  AND source_relation."external_catalog_name" = 'bmk'
  AND (
        {distinct_checks_sql}
  )
""".strip()


def build_bmk_parser_sync_relation_summary_sql(target_relation_name: str) -> str:
    # Собираем диагностическую сводку по relation после sync-а parser-derived полей.
    return f"""
SELECT
    COUNT(*) FILTER (
        WHERE "spectral_class" = 'O'
          AND "raw_sptype" ~ '^(OB|O/B|O[0-9]+(\\.[0-9]+)?/B)'
    ) AS ambiguous_ob_rows,
    COUNT(*) FILTER (WHERE "spectral_class" = 'OB') AS ob_rows,
    COUNT(*) FILTER (WHERE "spectral_class" = 'O') AS o_rows
FROM {quote_relation_name(target_relation_name, validate_identifiers=True)}
""".strip()


def build_bmk_training_summary_refresh_sql(
    relation_name: str = B_MK_TRAINING_SUMMARY_RELATION_NAME,
) -> tuple[str, str]:
    # Пересобираем компактную summary relation для training_reference.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    "spectral_class",
    COUNT(*) AS "row_count",
    COUNT(*) FILTER (WHERE "spectral_subclass" IS NOT NULL) AS "subclass_rows",
    COUNT(*) FILTER (WHERE "luminosity_class" IS NOT NULL) AS "luminosity_rows",
    COUNT(*) FILTER (WHERE "has_core_features" IS TRUE) AS "core_feature_rows",
    COUNT(*) FILTER (WHERE "has_flame_features" IS TRUE) AS "flame_feature_rows",
    COUNT(*) FILTER (WHERE "label_parse_status" = 'parsed') AS "parsed_rows",
    COUNT(*) FILTER (WHERE "label_parse_status" = 'partial') AS "partial_rows",
    NOW() AS "materialized_at_utc"
FROM {quote_relation_name(B_MK_TRAINING_REFERENCE_RELATION_NAME, validate_identifiers=True)}
GROUP BY "spectral_class"
ORDER BY "spectral_class" ASC
""".strip(),
    )


def build_bmk_quality_summary_refresh_sql(
    relation_name: str = B_MK_QUALITY_SUMMARY_RELATION_NAME,
) -> tuple[str, str]:
    # Пересобираем compact summary relation для quality-gated слоя.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    "quality_state",
    "ood_state",
    COUNT(*) AS "row_count",
    NOW() AS "materialized_at_utc"
FROM {quote_relation_name(B_MK_QUALITY_GATED_RELATION_NAME, validate_identifiers=True)}
GROUP BY "quality_state", "ood_state"
ORDER BY "quality_state" ASC, "ood_state" ASC
""".strip(),
    )


def build_bmk_unknown_summary_refresh_sql(
    relation_name: str = B_MK_UNKNOWN_SUMMARY_RELATION_NAME,
) -> tuple[str, str]:
    # Пересобираем compact summary relation для unknown/review слоя.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    "review_bucket",
    COUNT(*) AS "row_count",
    NOW() AS "materialized_at_utc"
FROM {quote_relation_name(B_MK_UNKNOWN_REVIEW_RELATION_NAME, validate_identifiers=True)}
GROUP BY "review_bucket"
ORDER BY "review_bucket" ASC NULLS LAST
""".strip(),
    )


__all__ = [
    "build_bmk_parser_sync_relation_summary_sql",
    "build_bmk_parser_sync_update_sql",
    "build_bmk_quality_summary_refresh_sql",
    "build_bmk_training_summary_refresh_sql",
    "build_bmk_unknown_summary_refresh_sql",
]
