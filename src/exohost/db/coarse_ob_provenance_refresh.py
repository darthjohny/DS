# Файл `coarse_ob_provenance_refresh.py` слоя `db`.
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

GAIA_MK_QUALITY_GATED_RELATION_NAME = "lab.gaia_mk_quality_gated"
GAIA_OB_HOT_PROVENANCE_AUDIT_SOURCE_RELATION_NAME = "lab.gaia_ob_hot_provenance_audit_source"
GAIA_OB_HOT_PROVENANCE_AUDIT_CLEAN_RELATION_NAME = "public.gaia_ob_hot_provenance_audit_clean"
GAIA_OB_HOT_PROVENANCE_AUDIT_SUMMARY_RELATION_NAME = "lab.gaia_ob_hot_provenance_audit_summary"
GAIA_OB_HOT_PROVENANCE_CROSSWALK_SUMMARY_RELATION_NAME = (
    "lab.gaia_ob_hot_provenance_crosswalk_summary"
)

GAIA_OB_HOT_PROVENANCE_LOCAL_COLUMNS: tuple[str, ...] = (
    "spectral_class",
    "spectral_subclass",
    "luminosity_class",
    "quality_state",
    "quality_reason",
    "review_bucket",
    "ood_state",
    "ood_reason",
    "random_index",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_flame",
    "radius_gspphot",
    "lum_flame",
    "evolstage_flame",
    "non_single_star",
    "classprob_dsc_combmod_star",
    "quality_gate_version",
    "quality_gated_at_utc",
)


class DbapiCursorProtocol(Protocol):
    # Минимальный DB-API cursor contract, который реально используем в refresh-helper.
    def execute(self, operation: str) -> object: ...

    def fetchone(self) -> tuple[object, ...] | None: ...


@dataclass(frozen=True, slots=True)
class CoarseObProvenanceRefreshSummary:
    # Сводка refresh-а локального provenance audit слоя.
    source_relation_name: str
    source_rows_loaded: int
    audit_clean_rows_updated: int
    summary_rows_loaded: int
    crosswalk_rows_loaded: int


def build_gaia_ob_hot_provenance_source_refresh_sql(
    relation_name: str = GAIA_OB_HOT_PROVENANCE_AUDIT_SOURCE_RELATION_NAME,
) -> tuple[str, str]:
    # Пересобираем локальный hot O/B/OB pass-pool из актуального quality-gated слоя.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    source_relation_sql = quote_relation_name(
        GAIA_MK_QUALITY_GATED_RELATION_NAME,
        validate_identifiers=True,
    )
    local_columns_sql = ",\n    ".join(f'"{column_name}"' for column_name in ("source_id", *GAIA_OB_HOT_PROVENANCE_LOCAL_COLUMNS))
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    {local_columns_sql}
FROM {source_relation_sql}
WHERE "quality_state" = 'pass'
  AND "spectral_class" IN ('O', 'B', 'OB')
  AND "teff_gspphot" >= 10000
ORDER BY "random_index" ASC NULLS LAST, "source_id" ASC
""".strip(),
    )


def build_gaia_ob_hot_audit_clean_sync_sql(
    relation_name: str = GAIA_OB_HOT_PROVENANCE_AUDIT_CLEAN_RELATION_NAME,
    *,
    source_relation_name: str = GAIA_OB_HOT_PROVENANCE_AUDIT_SOURCE_RELATION_NAME,
) -> str:
    # Синхронизируем local label/physics поля в Gaia audit clean relation без нового похода в Gaia.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    source_relation_sql = quote_relation_name(
        source_relation_name,
        validate_identifiers=True,
    )
    assignments_sql = ",\n    ".join(
        f'"{column_name}" = { _build_source_expression(column_name) }'
        for column_name in GAIA_OB_HOT_PROVENANCE_LOCAL_COLUMNS
    )
    distinct_checks_sql = " OR\n        ".join(
        'audit_clean.'
        f'"{column_name}" IS DISTINCT FROM { _build_source_expression(column_name) }'
        for column_name in GAIA_OB_HOT_PROVENANCE_LOCAL_COLUMNS
    )
    return f"""
UPDATE {relation_sql} AS audit_clean
SET
    {assignments_sql}
FROM {source_relation_sql} AS source_relation
WHERE audit_clean."source_id" = source_relation."source_id"
  AND (
        {distinct_checks_sql}
  )
""".strip()


def build_gaia_ob_hot_provenance_summary_refresh_sql(
    relation_name: str = GAIA_OB_HOT_PROVENANCE_AUDIT_SUMMARY_RELATION_NAME,
) -> tuple[str, str]:
    # Пересобираем compact summary по локальным классам на обновленном Gaia audit clean.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    source_relation_sql = quote_relation_name(
        GAIA_OB_HOT_PROVENANCE_AUDIT_CLEAN_RELATION_NAME,
        validate_identifiers=True,
    )
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
SELECT
    "spectral_class",
    COUNT(*) AS "n_rows",
    COUNT(*) FILTER (WHERE "in_gold_sample_oba_stars" IS TRUE) AS "n_in_gold_sample_oba_stars",
    COUNT(*) FILTER (WHERE "spectraltype_esphs" IS NOT NULL) AS "n_with_esphs_type",
    COUNT(*) FILTER (WHERE "esphs_class_letter" = 'O') AS "n_esphs_o",
    COUNT(*) FILTER (WHERE "esphs_class_letter" = 'B') AS "n_esphs_b",
    COUNT(*) FILTER (WHERE "esphs_class_letter" = 'A') AS "n_esphs_a",
    AVG(CASE WHEN "in_gold_sample_oba_stars" IS TRUE THEN 1.0 ELSE 0.0 END) AS "gold_sample_share",
    AVG(CASE WHEN "esphs_class_letter" = 'O' THEN 1.0 ELSE 0.0 END) AS "esphs_o_share",
    AVG(CASE WHEN "esphs_class_letter" = 'B' THEN 1.0 ELSE 0.0 END) AS "esphs_b_share",
    AVG(CASE WHEN "esphs_class_letter" = 'A' THEN 1.0 ELSE 0.0 END) AS "esphs_a_share",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "teff_gspphot") AS "median_teff_gspphot",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "teff_esphs") AS "median_teff_esphs",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "radius_flame") AS "median_radius_flame",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "radius_gspphot") AS "median_radius_gspphot"
FROM {source_relation_sql}
GROUP BY "spectral_class"
ORDER BY "spectral_class" ASC
""".strip(),
    )


def build_gaia_ob_hot_crosswalk_summary_refresh_sql(
    relation_name: str = GAIA_OB_HOT_PROVENANCE_CROSSWALK_SUMMARY_RELATION_NAME,
) -> tuple[str, str]:
    # Пересобираем local-vs-ESP-HS crosswalk после sync-а локальных классов.
    relation_sql = quote_relation_name(relation_name, validate_identifiers=True)
    source_relation_sql = quote_relation_name(
        GAIA_OB_HOT_PROVENANCE_AUDIT_CLEAN_RELATION_NAME,
        validate_identifiers=True,
    )
    return (
        f"DROP TABLE IF EXISTS {relation_sql}",
        f"""
CREATE TABLE {relation_sql} AS
WITH local_totals AS (
    SELECT
        "spectral_class",
        COUNT(*) AS "n_local_class_rows"
    FROM {source_relation_sql}
    GROUP BY "spectral_class"
)
SELECT
    audit_clean."spectral_class",
    audit_clean."esphs_class_letter",
    COUNT(*) AS "n_rows",
    local_totals."n_local_class_rows",
    COUNT(*)::DOUBLE PRECISION / NULLIF(local_totals."n_local_class_rows", 0)::DOUBLE PRECISION AS "share_within_local_class",
    COUNT(*) FILTER (WHERE audit_clean."in_gold_sample_oba_stars" IS TRUE) AS "n_in_gold_sample_oba_stars",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY audit_clean."teff_gspphot") AS "median_teff_gspphot",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY audit_clean."teff_esphs") AS "median_teff_esphs",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY audit_clean."radius_flame") AS "median_radius_flame"
FROM {source_relation_sql} AS audit_clean
INNER JOIN local_totals
    ON local_totals."spectral_class" = audit_clean."spectral_class"
GROUP BY
    audit_clean."spectral_class",
    audit_clean."esphs_class_letter",
    local_totals."n_local_class_rows"
ORDER BY audit_clean."spectral_class" ASC, audit_clean."esphs_class_letter" ASC NULLS LAST
""".strip(),
    )


def refresh_gaia_ob_hot_provenance_relations(
    engine: Engine,
) -> CoarseObProvenanceRefreshSummary:
    # Пересобираем local provenance source и summary relations после parser sync.
    _validate_gaia_ob_provenance_columns(engine)
    dbapi_connection = engine.raw_connection()
    cursor = dbapi_connection.cursor()
    try:
        drop_sql, create_sql = build_gaia_ob_hot_provenance_source_refresh_sql()
        cursor.execute(drop_sql)
        cursor.execute(create_sql)
        cursor.execute(build_gaia_ob_hot_audit_clean_sync_sql())
        audit_clean_rows_updated = max(int(cursor.rowcount), 0)
        for drop_sql, create_sql in (
            build_gaia_ob_hot_provenance_summary_refresh_sql(),
            build_gaia_ob_hot_crosswalk_summary_refresh_sql(),
        ):
            cursor.execute(drop_sql)
            cursor.execute(create_sql)
        cursor.execute(
            f'SELECT COUNT(*) FROM {quote_relation_name(GAIA_OB_HOT_PROVENANCE_AUDIT_SOURCE_RELATION_NAME, validate_identifiers=True)}'
        )
        source_rows_loaded = _scalar_to_int(cursor.fetchone(), relation_name=GAIA_OB_HOT_PROVENANCE_AUDIT_SOURCE_RELATION_NAME)
        cursor.execute(
            f'SELECT COUNT(*) FROM {quote_relation_name(GAIA_OB_HOT_PROVENANCE_AUDIT_SUMMARY_RELATION_NAME, validate_identifiers=True)}'
        )
        summary_rows_loaded = _scalar_to_int(cursor.fetchone(), relation_name=GAIA_OB_HOT_PROVENANCE_AUDIT_SUMMARY_RELATION_NAME)
        cursor.execute(
            f'SELECT COUNT(*) FROM {quote_relation_name(GAIA_OB_HOT_PROVENANCE_CROSSWALK_SUMMARY_RELATION_NAME, validate_identifiers=True)}'
        )
        crosswalk_rows_loaded = _scalar_to_int(cursor.fetchone(), relation_name=GAIA_OB_HOT_PROVENANCE_CROSSWALK_SUMMARY_RELATION_NAME)
        dbapi_connection.commit()
        return CoarseObProvenanceRefreshSummary(
            source_relation_name=GAIA_OB_HOT_PROVENANCE_AUDIT_SOURCE_RELATION_NAME,
            source_rows_loaded=source_rows_loaded,
            audit_clean_rows_updated=audit_clean_rows_updated,
            summary_rows_loaded=summary_rows_loaded,
            crosswalk_rows_loaded=crosswalk_rows_loaded,
        )
    except Exception:
        dbapi_connection.rollback()
        raise
    finally:
        cursor.close()
        dbapi_connection.close()


def _validate_gaia_ob_provenance_columns(engine: Engine) -> None:
    required_quality_columns = ("source_id", *GAIA_OB_HOT_PROVENANCE_LOCAL_COLUMNS)
    quality_columns = set(
        relation_columns(
            engine,
            GAIA_MK_QUALITY_GATED_RELATION_NAME,
            validate_identifiers=True,
        )
    )
    missing_quality_columns = tuple(
        column_name
        for column_name in required_quality_columns
        if column_name not in quality_columns
    )
    if missing_quality_columns:
        missing_columns_sql = ", ".join(missing_quality_columns)
        raise RuntimeError(
            "Gaia O/B provenance source is missing required quality-gated columns: "
            f"{missing_columns_sql}"
        )

    clean_columns = set(
        relation_columns(
            engine,
            GAIA_OB_HOT_PROVENANCE_AUDIT_CLEAN_RELATION_NAME,
            validate_identifiers=True,
        )
    )
    missing_clean_columns = tuple(
        column_name
        for column_name in ("source_id", *GAIA_OB_HOT_PROVENANCE_LOCAL_COLUMNS)
        if column_name not in clean_columns
    )
    if missing_clean_columns:
        missing_columns_sql = ", ".join(missing_clean_columns)
        raise RuntimeError(
            "Gaia O/B provenance clean relation is missing required columns: "
            f"{missing_columns_sql}"
        )


def _scalar_to_int(
    row: tuple[object, ...] | None,
    *,
    relation_name: str,
) -> int:
    if row is None:
        raise RuntimeError(
            "Gaia O/B provenance summary query returned no rows: "
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
        "Unable to convert provenance scalar to int: "
        f"{relation_name} -> {value!r}"
    )


def _build_source_expression(column_name: str) -> str:
    if column_name == "quality_gated_at_utc":
        return 'source_relation."quality_gated_at_utc"::TEXT'
    return f'source_relation."{column_name}"'
