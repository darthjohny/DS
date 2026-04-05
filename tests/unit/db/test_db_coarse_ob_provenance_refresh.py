# Тестовый файл `test_db_coarse_ob_provenance_refresh.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.db.coarse_ob_provenance_refresh import (
    build_gaia_ob_hot_audit_clean_sync_sql,
    build_gaia_ob_hot_crosswalk_summary_refresh_sql,
    build_gaia_ob_hot_provenance_source_refresh_sql,
    build_gaia_ob_hot_provenance_summary_refresh_sql,
)


def test_build_gaia_ob_hot_provenance_source_refresh_sql_uses_ob_boundary_pool() -> None:
    drop_sql, create_sql = build_gaia_ob_hot_provenance_source_refresh_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_ob_hot_provenance_audit_source"' in drop_sql
    assert 'FROM "lab"."gaia_mk_quality_gated"' in create_sql
    assert '"spectral_class" IN (\'O\', \'B\', \'OB\')' in create_sql
    assert '"quality_state" = \'pass\'' in create_sql


def test_build_gaia_ob_hot_audit_clean_sync_sql_syncs_local_columns_by_source_id() -> None:
    sql = build_gaia_ob_hot_audit_clean_sync_sql()

    assert 'UPDATE "public"."gaia_ob_hot_provenance_audit_clean" AS audit_clean' in sql
    assert 'FROM "lab"."gaia_ob_hot_provenance_audit_source" AS source_relation' in sql
    assert 'audit_clean."source_id" = source_relation."source_id"' in sql
    assert '"spectral_class" = source_relation."spectral_class"' in sql


def test_build_gaia_ob_hot_provenance_summary_refresh_sql_uses_clean_relation() -> None:
    drop_sql, create_sql = build_gaia_ob_hot_provenance_summary_refresh_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_ob_hot_provenance_audit_summary"' in drop_sql
    assert 'FROM "public"."gaia_ob_hot_provenance_audit_clean"' in create_sql
    assert 'GROUP BY "spectral_class"' in create_sql


def test_build_gaia_ob_hot_crosswalk_summary_refresh_sql_uses_esphs_crosswalk() -> None:
    drop_sql, create_sql = build_gaia_ob_hot_crosswalk_summary_refresh_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_ob_hot_provenance_crosswalk_summary"' in drop_sql
    assert 'WITH local_totals AS (' in create_sql
    assert 'audit_clean."esphs_class_letter"' in create_sql
