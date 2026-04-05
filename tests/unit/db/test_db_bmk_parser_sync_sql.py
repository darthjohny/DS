# Тестовый файл `test_db_bmk_parser_sync_sql.py` домена `db`.
#
# Этот файл проверяет только:
# - проверку логики домена: relation-layer, materialization и SQL-helper;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `db` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.db.bmk_parser_sync_contracts import (
    B_MK_PARSER_DERIVED_COLUMNS,
    B_MK_PARSER_SYNC_SOURCE_RELATION_NAME,
    B_MK_QUALITY_GATED_RELATION_NAME,
)
from exohost.db.bmk_parser_sync_sql import (
    build_bmk_parser_sync_relation_summary_sql,
    build_bmk_parser_sync_update_sql,
    build_bmk_quality_summary_refresh_sql,
    build_bmk_training_summary_refresh_sql,
    build_bmk_unknown_summary_refresh_sql,
)


def test_build_bmk_parser_sync_update_sql_uses_join_keys_and_parser_fields() -> None:
    sql = build_bmk_parser_sync_update_sql(B_MK_QUALITY_GATED_RELATION_NAME)

    assert 'UPDATE "lab"."gaia_mk_quality_gated" AS target_relation' in sql
    assert (
        f'FROM "{B_MK_PARSER_SYNC_SOURCE_RELATION_NAME.split(".")[0]}"."{B_MK_PARSER_SYNC_SOURCE_RELATION_NAME.split(".")[1]}" AS source_relation'
        in sql
    )
    assert 'target_relation."xmatch_batch_id" = source_relation."xmatch_batch_id"' in sql
    assert 'target_relation."source_id" = source_relation."source_id"' in sql
    assert (
        'target_relation."external_row_id" = source_relation."external_row_id"' in sql
    )
    for column_name in B_MK_PARSER_DERIVED_COLUMNS:
        assert f'"{column_name}" = source_relation."{column_name}"' in sql


def test_build_bmk_parser_sync_relation_summary_sql_counts_ob_and_o_rows() -> None:
    sql = build_bmk_parser_sync_relation_summary_sql(B_MK_QUALITY_GATED_RELATION_NAME)

    assert 'COUNT(*) FILTER (' in sql
    assert '"spectral_class" = \'OB\'' in sql
    assert '"spectral_class" = \'O\'' in sql
    assert '"raw_sptype" ~' in sql
    assert 'FROM "lab"."gaia_mk_quality_gated"' in sql


def test_build_bmk_training_summary_refresh_sql_targets_summary_relation() -> None:
    drop_sql, create_sql = build_bmk_training_summary_refresh_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_mk_training_reference_summary"' in drop_sql
    assert 'FROM "lab"."gaia_mk_training_reference"' in create_sql
    assert 'GROUP BY "spectral_class"' in create_sql


def test_build_bmk_quality_summary_refresh_sql_targets_quality_relation() -> None:
    drop_sql, create_sql = build_bmk_quality_summary_refresh_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_mk_quality_gated_summary"' in drop_sql
    assert 'FROM "lab"."gaia_mk_quality_gated"' in create_sql
    assert 'GROUP BY "quality_state", "ood_state"' in create_sql


def test_build_bmk_unknown_summary_refresh_sql_targets_unknown_relation() -> None:
    drop_sql, create_sql = build_bmk_unknown_summary_refresh_sql()

    assert 'DROP TABLE IF EXISTS "lab"."gaia_mk_unknown_review_summary"' in drop_sql
    assert 'FROM "lab"."gaia_mk_unknown_review"' in create_sql
    assert 'GROUP BY "review_bucket"' in create_sql
