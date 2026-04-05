# Файл `refinement_family_views.py` слоя `db`.
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

from exohost.contracts.refinement_family_dataset_contracts import (
    REFINEMENT_ENABLED_SPECTRAL_CLASSES,
    build_refinement_family_view_name,
    validate_refinement_family_class,
)
from exohost.db.relations import (
    quote_identifier,
    quote_relation_name,
    relation_columns,
    split_relation_name,
)

GAIA_MK_REFINEMENT_BASE_VIEW_RELATION_NAME = "lab.v_gaia_mk_refinement_training"
GAIA_MK_REFINEMENT_SUPPORT_AUDIT_RELATION_NAME = "lab.gaia_mk_refinement_support_audit"
GAIA_MK_REFINEMENT_SUPPORT_KEEP_COLUMN = "keep_ge_15"

GAIA_MK_REFINEMENT_FAMILY_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "source_id",
    "spectral_class",
    "spectral_subclass",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "radius_flame",
    "lum_flame",
    "evolstage_flame",
    "phot_g_mean_mag",
    "quality_state",
    "ood_state",
)

GAIA_MK_REFINEMENT_SUPPORT_AUDIT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "spectral_class",
    "full_subclass",
    GAIA_MK_REFINEMENT_SUPPORT_KEEP_COLUMN,
)


class DbapiCursorProtocol(Protocol):
    # Минимальный DB-API cursor contract, который реально используем в модуле.
    def execute(self, operation: str) -> object: ...

    def fetchone(self) -> tuple[object, ...] | None: ...


@dataclass(frozen=True, slots=True)
class RefinementFamilyViewSummary:
    # Фактическая row/cardinality summary одной materialized family-view.
    spectral_class: str
    relation_name: str
    rows_loaded: int
    distinct_source_ids: int
    distinct_subclasses: int


def build_refinement_family_view_sql(
    spectral_class: str,
    *,
    source_relation_name: str = GAIA_MK_REFINEMENT_BASE_VIEW_RELATION_NAME,
    support_audit_relation_name: str = GAIA_MK_REFINEMENT_SUPPORT_AUDIT_RELATION_NAME,
) -> str:
    # Собираем `CREATE OR REPLACE VIEW` для одной family relation.
    normalized_class = validate_refinement_family_class(spectral_class)
    target_relation_name = build_refinement_family_view_name(normalized_class)
    target_relation_sql = quote_relation_name(
        target_relation_name,
        validate_identifiers=True,
    )
    source_relation_sql = quote_relation_name(
        source_relation_name,
        validate_identifiers=True,
    )
    support_relation_sql = quote_relation_name(
        support_audit_relation_name,
        validate_identifiers=True,
    )
    return f"""
CREATE OR REPLACE VIEW {target_relation_sql} AS
SELECT
    source_view.*,
    source_view."spectral_class" || source_view."spectral_subclass"::TEXT AS full_subclass_label
FROM {source_relation_sql} AS source_view
INNER JOIN {support_relation_sql} AS support_audit
    ON support_audit."spectral_class" = source_view."spectral_class"
   AND support_audit."full_subclass" = source_view."spectral_class" || source_view."spectral_subclass"::TEXT
WHERE source_view."spectral_class" = '{normalized_class}'
  AND source_view."quality_state" = 'pass'
  AND source_view."ood_state" = 'in_domain'
  AND source_view."teff_gspphot" IS NOT NULL
  AND source_view."logg_gspphot" IS NOT NULL
  AND source_view."mh_gspphot" IS NOT NULL
  AND source_view."bp_rp" IS NOT NULL
  AND source_view."parallax" IS NOT NULL
  AND source_view."parallax_over_error" IS NOT NULL
  AND source_view."ruwe" IS NOT NULL
  AND source_view."radius_flame" IS NOT NULL
  AND source_view."lum_flame" IS NOT NULL
  AND source_view."evolstage_flame" IS NOT NULL
  AND source_view."phot_g_mean_mag" IS NOT NULL
  AND support_audit."{GAIA_MK_REFINEMENT_SUPPORT_KEEP_COLUMN}" = TRUE
ORDER BY source_view."source_id" ASC
""".strip()


def build_refinement_family_schema_sql() -> str:
    # Гарантируем наличие schema `lab` до materialization views.
    schema_name, _ = split_relation_name(
        GAIA_MK_REFINEMENT_BASE_VIEW_RELATION_NAME,
        validate_identifiers=True,
    )
    return f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema_name)}"


def materialize_refinement_family_views(
    engine: Engine,
    *,
    source_relation_name: str = GAIA_MK_REFINEMENT_BASE_VIEW_RELATION_NAME,
    support_audit_relation_name: str = GAIA_MK_REFINEMENT_SUPPORT_AUDIT_RELATION_NAME,
) -> tuple[RefinementFamilyViewSummary, ...]:
    # Создаем/обновляем все second-wave refinement family views.
    _validate_required_columns(
        engine,
        source_relation_name=source_relation_name,
        support_audit_relation_name=support_audit_relation_name,
    )
    dbapi_connection = engine.raw_connection()
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(build_refinement_family_schema_sql())
        for spectral_class in REFINEMENT_ENABLED_SPECTRAL_CLASSES:
            cursor.execute(
                build_refinement_family_view_sql(
                    spectral_class,
                    source_relation_name=source_relation_name,
                    support_audit_relation_name=support_audit_relation_name,
                )
            )
        dbapi_connection.commit()

        return tuple(
            _build_family_view_summary(
                cursor,
                spectral_class=spectral_class,
            )
            for spectral_class in REFINEMENT_ENABLED_SPECTRAL_CLASSES
        )
    except Exception:
        dbapi_connection.rollback()
        raise
    finally:
        cursor.close()
        dbapi_connection.close()


def _validate_required_columns(
    engine: Engine,
    *,
    source_relation_name: str,
    support_audit_relation_name: str,
) -> None:
    source_columns = set(
        relation_columns(
            engine,
            source_relation_name,
            validate_identifiers=True,
        )
    )
    missing_source_columns = tuple(
        column_name
        for column_name in GAIA_MK_REFINEMENT_FAMILY_REQUIRED_SOURCE_COLUMNS
        if column_name not in source_columns
    )
    if missing_source_columns:
        missing_columns_sql = ", ".join(missing_source_columns)
        raise RuntimeError(
            "Refinement family source is missing required columns: "
            f"{missing_columns_sql}"
        )

    audit_columns = set(
        relation_columns(
            engine,
            support_audit_relation_name,
            validate_identifiers=True,
        )
    )
    missing_audit_columns = tuple(
        column_name
        for column_name in GAIA_MK_REFINEMENT_SUPPORT_AUDIT_REQUIRED_COLUMNS
        if column_name not in audit_columns
    )
    if missing_audit_columns:
        missing_columns_sql = ", ".join(missing_audit_columns)
        raise RuntimeError(
            "Refinement family support audit is missing required columns: "
            f"{missing_columns_sql}"
        )


def _build_family_view_summary(
    cursor: DbapiCursorProtocol,
    *,
    spectral_class: str,
) -> RefinementFamilyViewSummary:
    normalized_class = validate_refinement_family_class(spectral_class)
    relation_name = build_refinement_family_view_name(normalized_class)
    relation_sql = quote_relation_name(
        relation_name,
        validate_identifiers=True,
    )
    cursor.execute(
        f"""
        SELECT
            COUNT(*) AS rows_loaded,
            COUNT(DISTINCT "source_id") AS distinct_source_ids,
            COUNT(DISTINCT "spectral_subclass") AS distinct_subclasses
        FROM {relation_sql}
        """
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(
            "Refinement family view summary query returned no rows: "
            f"{relation_name}"
        )
    return RefinementFamilyViewSummary(
        spectral_class=normalized_class,
        relation_name=relation_name,
        rows_loaded=_to_int_count(row[0], relation_name=relation_name),
        distinct_source_ids=_to_int_count(row[1], relation_name=relation_name),
        distinct_subclasses=_to_int_count(row[2], relation_name=relation_name),
    )


def _to_int_count(value: object, *, relation_name: str) -> int:
    # Явно сужаем DB-aggregate scalar до int, чтобы не тащить Any в summary layer.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value:
            return int(stripped_value)
    raise RuntimeError(
        "Refinement family view summary contains non-integer aggregate value "
        f"for relation {relation_name}: {value!r}"
    )
