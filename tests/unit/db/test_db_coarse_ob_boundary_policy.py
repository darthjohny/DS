# Тестовый файл `test_db_coarse_ob_boundary_policy.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.db.coarse_ob_boundary_policy import (
    GAIA_OB_POLICY_VERSION,
    build_ob_boundary_policy_summary_sql,
    build_ob_boundary_subset_sql,
    build_secure_o_like_subset_sql,
)


def test_build_secure_o_like_subset_sql_uses_esphs_o_non999_policy() -> None:
    drop_sql, create_sql = build_secure_o_like_subset_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_ob_secure_o_like_subset"' in drop_sql
    assert 'FROM "public"."gaia_ob_hot_provenance_audit_clean" AS source_relation' in create_sql
    assert 'source_relation."spectral_class" IN (\'O\', \'OB\')' in create_sql
    assert 'source_relation."esphs_class_letter" = \'O\'' in create_sql
    assert 'COALESCE(source_relation."flags_esphs", -1) <> 999' in create_sql
    assert f"'{GAIA_OB_POLICY_VERSION}' AS \"ob_policy_version\"" in create_sql


def test_build_ob_boundary_subset_sql_excludes_secure_o_like_rows() -> None:
    drop_sql, create_sql = build_ob_boundary_subset_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_ob_boundary_subset"' in drop_sql
    assert 'NOT (' in create_sql
    assert 'source_relation."spectral_class" = \'OB\'' in create_sql
    assert 'explicit_o_not_confirmed_by_esphs' in create_sql


def test_build_ob_boundary_policy_summary_sql_aggregates_both_subsets() -> None:
    drop_sql, create_sql = build_ob_boundary_policy_summary_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_ob_boundary_policy_summary"' in drop_sql
    assert 'FROM "lab"."gaia_ob_secure_o_like_subset" AS secure_subset' in create_sql
    assert 'FROM "lab"."gaia_ob_boundary_subset" AS boundary_subset' in create_sql
    assert 'UNION ALL' in create_sql
