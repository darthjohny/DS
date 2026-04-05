# Файл `coarse_ob_review_pool.py` слоя `db`.
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

from exohost.db.coarse_ob_boundary_policy import GAIA_OB_BOUNDARY_RELATION_NAME
from exohost.db.relations import quote_relation_name, relation_columns

GAIA_OB_REVIEW_POOL_SOURCE_RELATION_NAME = GAIA_OB_BOUNDARY_RELATION_NAME
GAIA_OB_REVIEW_POOL_RELATION_NAME = "lab.gaia_ob_boundary_review"
GAIA_OB_REVIEW_POOL_SUMMARY_RELATION_NAME = "lab.gaia_ob_boundary_review_summary"

GAIA_OB_REVIEW_POOL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_id",
    "spectral_class",
    "ob_policy_bucket",
    "ob_policy_reason",
    "esphs_class_letter",
)


class DbapiCursorProtocol(Protocol):
    # Минимальный DB-API cursor contract для review-pool materialization.
    def execute(self, operation: str) -> object: ...

    def fetchone(self) -> tuple[object, ...] | None: ...


@dataclass(frozen=True, slots=True)
class CoarseObReviewPoolSummary:
    # Сводка materialized O/B review-pool relations.
    source_relation_name: str
    review_rows_loaded: int
    summary_rows_loaded: int


def build_gaia_ob_boundary_review_sql(
    relation_name: str = GAIA_OB_REVIEW_POOL_RELATION_NAME,
    *,
    source_relation_name: str = GAIA_OB_REVIEW_POOL_SOURCE_RELATION_NAME,
) -> tuple[str, str]:
    # Собираем явный review-pool из already materialized O/B boundary subset.
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
    'review' AS "ob_review_state",
    source_relation."ob_policy_reason" AS "ob_review_reason"
FROM {source_relation_sql} AS source_relation
ORDER BY source_relation."source_id" ASC
""".strip(),
    )


def build_gaia_ob_boundary_review_summary_sql(
    relation_name: str = GAIA_OB_REVIEW_POOL_SUMMARY_RELATION_NAME,
) -> tuple[str, str]:
    # Пересобираем compact summary по O/B review-pool.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    source_relation_sql = quote_relation_name(
        GAIA_OB_REVIEW_POOL_RELATION_NAME,
        validate_identifiers=True,
    )
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    "spectral_class" AS "local_spectral_class",
    "ob_review_reason",
    "esphs_class_letter",
    COUNT(*) AS "n_rows",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "teff_esphs") AS "median_teff_esphs",
    NOW() AS "materialized_at_utc"
FROM {source_relation_sql}
GROUP BY "spectral_class", "ob_review_reason", "esphs_class_letter"
ORDER BY "spectral_class" ASC, "ob_review_reason" ASC, "esphs_class_letter" ASC NULLS LAST
""".strip(),
    )


def materialize_coarse_ob_review_pool_relations(
    engine: Engine,
    *,
    source_relation_name: str = GAIA_OB_REVIEW_POOL_SOURCE_RELATION_NAME,
) -> CoarseObReviewPoolSummary:
    # Materialize явный review-pool для hot O/B boundary cases.
    _validate_review_pool_source_columns(
        engine,
        source_relation_name=source_relation_name,
    )
    dbapi_connection = engine.raw_connection()
    cursor = dbapi_connection.cursor()
    try:
        for drop_sql, create_sql in (
            build_gaia_ob_boundary_review_sql(
                source_relation_name=source_relation_name
            ),
            build_gaia_ob_boundary_review_summary_sql(),
        ):
            cursor.execute(drop_sql)
            cursor.execute(create_sql)
        cursor.execute(
            f'SELECT COUNT(*) FROM {quote_relation_name(GAIA_OB_REVIEW_POOL_RELATION_NAME, validate_identifiers=True)}'
        )
        review_rows_loaded = _scalar_to_int(
            cursor.fetchone(),
            relation_name=GAIA_OB_REVIEW_POOL_RELATION_NAME,
        )
        cursor.execute(
            f'SELECT COUNT(*) FROM {quote_relation_name(GAIA_OB_REVIEW_POOL_SUMMARY_RELATION_NAME, validate_identifiers=True)}'
        )
        summary_rows_loaded = _scalar_to_int(
            cursor.fetchone(),
            relation_name=GAIA_OB_REVIEW_POOL_SUMMARY_RELATION_NAME,
        )
        dbapi_connection.commit()
        return CoarseObReviewPoolSummary(
            source_relation_name=source_relation_name,
            review_rows_loaded=review_rows_loaded,
            summary_rows_loaded=summary_rows_loaded,
        )
    except Exception:
        dbapi_connection.rollback()
        raise
    finally:
        cursor.close()
        dbapi_connection.close()


def _validate_review_pool_source_columns(
    engine: Engine,
    *,
    source_relation_name: str,
) -> None:
    source_columns = set(
        relation_columns(engine, source_relation_name, validate_identifiers=True)
    )
    missing_columns = tuple(
        column_name
        for column_name in GAIA_OB_REVIEW_POOL_REQUIRED_COLUMNS
        if column_name not in source_columns
    )
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise RuntimeError(
            "Coarse O/B review-pool source is missing required columns: "
            f"{missing_columns_sql}"
        )


def _scalar_to_int(
    row: tuple[object, ...] | None,
    *,
    relation_name: str,
) -> int:
    if row is None:
        raise RuntimeError(
            "Coarse O/B review-pool summary query returned no rows: "
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
        "Unable to convert coarse O/B review-pool scalar to int: "
        f"{relation_name} -> {value!r}"
    )
