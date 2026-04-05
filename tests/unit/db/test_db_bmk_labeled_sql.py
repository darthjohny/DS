# Тестовый файл `test_db_bmk_labeled_sql.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.db.bmk_labeled import (
    B_MK_EXTERNAL_LABELED_RELATION_NAME,
    build_bmk_external_labeled_schema_sql,
    build_bmk_external_labeled_source_query,
    build_delete_bmk_external_labeled_batch_sql,
)


def test_build_bmk_external_labeled_schema_sql_contains_relation_definition() -> None:
    schema_sql = build_bmk_external_labeled_schema_sql()

    assert len(schema_sql) == 3
    assert "CREATE SCHEMA IF NOT EXISTS" in schema_sql[0]
    assert B_MK_EXTERNAL_LABELED_RELATION_NAME.split(".", 1)[1] in schema_sql[1]
    assert "peculiarity_suffix" in schema_sql[1]
    assert "has_source_conflict" in schema_sql[1]
    assert "source_conflict_count" in schema_sql[1]
    assert "CREATE INDEX IF NOT EXISTS" in schema_sql[2]


def test_build_delete_bmk_external_labeled_batch_sql_targets_one_batch() -> None:
    sql = build_delete_bmk_external_labeled_batch_sql(
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
    )

    assert 'DELETE FROM "lab"."gaia_mk_external_labeled"' in sql
    assert '"xmatch_batch_id" =' in sql
    assert "xmatch_bmk_gaia_dr3__2026_03_26" in sql


def test_build_bmk_external_labeled_source_query_uses_selected_crossmatch_join() -> None:
    query = build_bmk_external_labeled_source_query(
        xmatch_batch_id="xmatch_bmk_gaia_dr3__2026_03_26",
    )

    assert 'FROM "lab"."gaia_mk_external_crossmatch" AS c' in query
    assert 'INNER JOIN "lab"."gaia_mk_external_filtered" AS f' in query
    assert 'c."xmatch_selected" IS TRUE' in query
    assert 'COUNT(*) OVER (' in query
    assert 'PARTITION BY c."source_id"' in query
    assert 'ORDER BY c."external_row_id" ASC' in query
