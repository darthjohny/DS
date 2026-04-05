# Тестовый файл `test_db_coarse_ob_review_pool.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.db.coarse_ob_review_pool import (
    build_gaia_ob_boundary_review_sql,
    build_gaia_ob_boundary_review_summary_sql,
)


def test_build_gaia_ob_boundary_review_sql_uses_boundary_subset_as_source() -> None:
    drop_sql, create_sql = build_gaia_ob_boundary_review_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_ob_boundary_review"' in drop_sql
    assert 'FROM "lab"."gaia_ob_boundary_subset" AS source_relation' in create_sql
    assert "'review' AS \"ob_review_state\"" in create_sql
    assert 'source_relation."ob_policy_reason" AS "ob_review_reason"' in create_sql


def test_build_gaia_ob_boundary_review_summary_sql_aggregates_review_pool() -> None:
    drop_sql, create_sql = build_gaia_ob_boundary_review_summary_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_ob_boundary_review_summary"' in drop_sql
    assert 'FROM "lab"."gaia_ob_boundary_review"' in create_sql
    assert 'GROUP BY "spectral_class", "ob_review_reason", "esphs_class_letter"' in create_sql
