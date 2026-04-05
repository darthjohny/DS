# Тестовый файл `test_db_refinement_family_views.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.db.refinement_family_views import (
    GAIA_MK_REFINEMENT_BASE_VIEW_RELATION_NAME,
    GAIA_MK_REFINEMENT_SUPPORT_AUDIT_RELATION_NAME,
    build_refinement_family_schema_sql,
    build_refinement_family_view_sql,
)


def test_build_refinement_family_schema_sql_targets_lab_schema() -> None:
    assert build_refinement_family_schema_sql() == 'CREATE SCHEMA IF NOT EXISTS "lab"'


def test_build_refinement_family_view_sql_uses_base_view_and_support_audit() -> None:
    sql = build_refinement_family_view_sql("K")

    assert 'CREATE OR REPLACE VIEW "lab"."v_gaia_mk_refinement_training_k" AS' in sql
    assert 'FROM "lab"."v_gaia_mk_refinement_training" AS source_view' in sql
    assert (
        'INNER JOIN "lab"."gaia_mk_refinement_support_audit" AS support_audit' in sql
    )
    assert 'WHERE source_view."spectral_class" = \'K\'' in sql
    assert 'source_view."quality_state" = \'pass\'' in sql
    assert 'source_view."ood_state" = \'in_domain\'' in sql
    assert 'source_view."radius_flame" IS NOT NULL' in sql
    assert 'source_view."evolstage_flame" IS NOT NULL' in sql
    assert 'support_audit."keep_ge_15" = TRUE' in sql
    assert 'full_subclass_label' in sql


def test_build_refinement_family_view_sql_supports_custom_relations() -> None:
    sql = build_refinement_family_view_sql(
        "A",
        source_relation_name=GAIA_MK_REFINEMENT_BASE_VIEW_RELATION_NAME,
        support_audit_relation_name=GAIA_MK_REFINEMENT_SUPPORT_AUDIT_RELATION_NAME,
    )

    assert '"lab"."v_gaia_mk_refinement_training_a"' in sql
