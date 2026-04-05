# Файл `coarse_ob_boundary_policy.py` слоя `db`.
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
from typing import Protocol

from sqlalchemy.engine import Engine

from exohost.db.relations import quote_relation_name, relation_columns

GAIA_OB_POLICY_SOURCE_RELATION_NAME = "public.gaia_ob_hot_provenance_audit_clean"
GAIA_OB_SECURE_O_LIKE_RELATION_NAME = "lab.gaia_ob_secure_o_like_subset"
GAIA_OB_BOUNDARY_RELATION_NAME = "lab.gaia_ob_boundary_subset"
GAIA_OB_POLICY_SUMMARY_RELATION_NAME = "lab.gaia_ob_boundary_policy_summary"
GAIA_OB_POLICY_VERSION = "coarse_ob_boundary_policy_v1"

GAIA_OB_POLICY_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "source_id",
    "spectral_class",
    "esphs_class_letter",
    "flags_esphs",
    "teff_esphs",
)


class DbapiCursorProtocol(Protocol):
    # Минимальный DB-API cursor contract, который реально используем в policy-helper.
    def execute(self, operation: str) -> object: ...

    def fetchone(self) -> tuple[object, ...] | None: ...


@dataclass(frozen=True, slots=True)
class CoarseObBoundaryPolicySummary:
    # Фактическая сводка materialized O/B policy relations.
    source_relation_name: str
    secure_o_like_rows: int
    ob_boundary_rows: int
    summary_rows_loaded: int


def build_secure_o_like_subset_sql(
    relation_name: str = GAIA_OB_SECURE_O_LIKE_RELATION_NAME,
    *,
    source_relation_name: str = GAIA_OB_POLICY_SOURCE_RELATION_NAME,
) -> tuple[str, str]:
    # Собираем strict secure O-like subset на базе Gaia ESP-HS hot-star semantics.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    source_relation_sql = quote_relation_name(
        source_relation_name,
        validate_identifiers=True,
    )
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    source_relation.*,
    '{GAIA_OB_POLICY_VERSION}' AS "ob_policy_version",
    'secure_o_like' AS "ob_policy_bucket",
    'esphs_o_non999' AS "ob_policy_reason"
FROM {source_relation_sql} AS source_relation
WHERE source_relation."spectral_class" IN ('O', 'OB')
  AND source_relation."esphs_class_letter" = 'O'
  AND COALESCE(source_relation."flags_esphs", -1) <> 999
ORDER BY source_relation."source_id" ASC
""".strip(),
    )


def build_ob_boundary_subset_sql(
    relation_name: str = GAIA_OB_BOUNDARY_RELATION_NAME,
    *,
    source_relation_name: str = GAIA_OB_POLICY_SOURCE_RELATION_NAME,
) -> tuple[str, str]:
    # Собираем boundary subset из оставшегося local hot O/OB пула.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    source_relation_sql = quote_relation_name(
        source_relation_name,
        validate_identifiers=True,
    )
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    source_relation.*,
    '{GAIA_OB_POLICY_VERSION}' AS "ob_policy_version",
    'ob_boundary' AS "ob_policy_bucket",
    CASE
        WHEN source_relation."spectral_class" = 'OB' THEN 'ambiguous_ob_label'
        WHEN source_relation."spectral_class" = 'O' THEN 'explicit_o_not_confirmed_by_esphs'
        ELSE 'ob_boundary_fallback'
    END AS "ob_policy_reason"
FROM {source_relation_sql} AS source_relation
WHERE source_relation."spectral_class" IN ('O', 'OB')
  AND NOT (
      source_relation."esphs_class_letter" = 'O'
      AND COALESCE(source_relation."flags_esphs", -1) <> 999
  )
ORDER BY source_relation."source_id" ASC
""".strip(),
    )


def build_ob_boundary_policy_summary_sql(
    relation_name: str = GAIA_OB_POLICY_SUMMARY_RELATION_NAME,
) -> tuple[str, str]:
    # Пересобираем compact summary для secure-vs-boundary O/B policy.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    secure_relation_sql = quote_relation_name(
        GAIA_OB_SECURE_O_LIKE_RELATION_NAME,
        validate_identifiers=True,
    )
    boundary_relation_sql = quote_relation_name(
        GAIA_OB_BOUNDARY_RELATION_NAME,
        validate_identifiers=True,
    )
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    secure_subset."ob_policy_bucket",
    secure_subset."spectral_class" AS "local_spectral_class",
    COUNT(*) AS "n_rows",
    COUNT(*) FILTER (WHERE secure_subset."esphs_class_letter" = 'O') AS "n_esphs_o",
    COUNT(*) FILTER (WHERE secure_subset."esphs_class_letter" = 'B') AS "n_esphs_b",
    COUNT(*) FILTER (WHERE secure_subset."esphs_class_letter" = 'U') AS "n_esphs_u",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY secure_subset."teff_esphs") AS "median_teff_esphs",
    NOW() AS "materialized_at_utc"
FROM {secure_relation_sql} AS secure_subset
GROUP BY secure_subset."ob_policy_bucket", secure_subset."spectral_class"

UNION ALL

SELECT
    boundary_subset."ob_policy_bucket",
    boundary_subset."spectral_class" AS "local_spectral_class",
    COUNT(*) AS "n_rows",
    COUNT(*) FILTER (WHERE boundary_subset."esphs_class_letter" = 'O') AS "n_esphs_o",
    COUNT(*) FILTER (WHERE boundary_subset."esphs_class_letter" = 'B') AS "n_esphs_b",
    COUNT(*) FILTER (WHERE boundary_subset."esphs_class_letter" = 'U') AS "n_esphs_u",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY boundary_subset."teff_esphs") AS "median_teff_esphs",
    NOW() AS "materialized_at_utc"
FROM {boundary_relation_sql} AS boundary_subset
GROUP BY boundary_subset."ob_policy_bucket", boundary_subset."spectral_class"

ORDER BY "ob_policy_bucket" ASC, "local_spectral_class" ASC
""".strip(),
    )


def materialize_coarse_ob_boundary_policy_relations(
    engine: Engine,
    *,
    source_relation_name: str = GAIA_OB_POLICY_SOURCE_RELATION_NAME,
) -> CoarseObBoundaryPolicySummary:
    # Materialize secure O-like и O/B boundary subsets из hot-star provenance source.
    _validate_policy_source_columns(
        engine,
        source_relation_name=source_relation_name,
    )
    dbapi_connection = engine.raw_connection()
    cursor = dbapi_connection.cursor()
    try:
        for drop_sql, create_sql in (
            build_secure_o_like_subset_sql(source_relation_name=source_relation_name),
            build_ob_boundary_subset_sql(source_relation_name=source_relation_name),
            build_ob_boundary_policy_summary_sql(),
        ):
            cursor.execute(drop_sql)
            cursor.execute(create_sql)
        cursor.execute(
            f'SELECT COUNT(*) FROM {quote_relation_name(GAIA_OB_SECURE_O_LIKE_RELATION_NAME, validate_identifiers=True)}'
        )
        secure_o_like_rows = _scalar_to_int(
            cursor.fetchone(),
            relation_name=GAIA_OB_SECURE_O_LIKE_RELATION_NAME,
        )
        cursor.execute(
            f'SELECT COUNT(*) FROM {quote_relation_name(GAIA_OB_BOUNDARY_RELATION_NAME, validate_identifiers=True)}'
        )
        ob_boundary_rows = _scalar_to_int(
            cursor.fetchone(),
            relation_name=GAIA_OB_BOUNDARY_RELATION_NAME,
        )
        cursor.execute(
            f'SELECT COUNT(*) FROM {quote_relation_name(GAIA_OB_POLICY_SUMMARY_RELATION_NAME, validate_identifiers=True)}'
        )
        summary_rows_loaded = _scalar_to_int(
            cursor.fetchone(),
            relation_name=GAIA_OB_POLICY_SUMMARY_RELATION_NAME,
        )
        dbapi_connection.commit()
        return CoarseObBoundaryPolicySummary(
            source_relation_name=source_relation_name,
            secure_o_like_rows=secure_o_like_rows,
            ob_boundary_rows=ob_boundary_rows,
            summary_rows_loaded=summary_rows_loaded,
        )
    except Exception:
        dbapi_connection.rollback()
        raise
    finally:
        cursor.close()
        dbapi_connection.close()


def _validate_policy_source_columns(
    engine: Engine,
    *,
    source_relation_name: str,
) -> None:
    source_columns = set(
        relation_columns(engine, source_relation_name, validate_identifiers=True)
    )
    missing_columns = tuple(
        column_name
        for column_name in GAIA_OB_POLICY_REQUIRED_SOURCE_COLUMNS
        if column_name not in source_columns
    )
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise RuntimeError(
            "Coarse O/B policy source is missing required columns: "
            f"{missing_columns_sql}"
        )


def _scalar_to_int(
    row: tuple[object, ...] | None,
    *,
    relation_name: str,
) -> int:
    if row is None:
        raise RuntimeError(
            "Coarse O/B policy summary query returned no rows: "
            f"{relation_name}"
        )
    value = row[0]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value.strip())
    raise RuntimeError(
        "Unable to convert coarse O/B policy scalar to int: "
        f"{relation_name} -> {value!r}"
    )
