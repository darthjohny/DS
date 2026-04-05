# Файл `bmk_parser_sync_execution.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from sqlalchemy.engine import Engine

from exohost.db.bmk_parser_sync_contracts import (
    B_MK_PARSER_SYNC_SOURCE_RELATION_NAME,
    B_MK_PARSER_SYNC_TARGET_RELATION_NAMES,
    BmkParserSyncRelationSummary,
    BmkParserSyncSummary,
    DbapiCursorProtocol,
)
from exohost.db.bmk_parser_sync_scalars import cursor_rowcount, scalar_to_int
from exohost.db.bmk_parser_sync_sql import (
    build_bmk_parser_sync_relation_summary_sql,
    build_bmk_parser_sync_update_sql,
    build_bmk_quality_summary_refresh_sql,
    build_bmk_training_summary_refresh_sql,
    build_bmk_unknown_summary_refresh_sql,
)
from exohost.db.bmk_parser_sync_validation import validate_bmk_parser_sync_columns


def sync_bmk_parser_fields_downstream(
    engine: Engine,
    *,
    source_relation_name: str = B_MK_PARSER_SYNC_SOURCE_RELATION_NAME,
) -> BmkParserSyncSummary:
    # Проталкиваем parser-derived поля из canonical labeled relation в downstream tables.
    validate_bmk_parser_sync_columns(
        engine,
        source_relation_name=source_relation_name,
    )
    dbapi_connection = engine.raw_connection()
    cursor = dbapi_connection.cursor()
    try:
        relation_summaries = tuple(
            _sync_target_relation(
                cursor,
                target_relation_name=target_relation_name,
                source_relation_name=source_relation_name,
            )
            for target_relation_name in B_MK_PARSER_SYNC_TARGET_RELATION_NAMES
        )
        _refresh_downstream_summary_tables(cursor)
        dbapi_connection.commit()
        return BmkParserSyncSummary(
            source_relation_name=source_relation_name,
            relation_summaries=relation_summaries,
        )
    except Exception:
        dbapi_connection.rollback()
        raise
    finally:
        cursor.close()
        dbapi_connection.close()


def _sync_target_relation(
    cursor: DbapiCursorProtocol,
    *,
    target_relation_name: str,
    source_relation_name: str,
) -> BmkParserSyncRelationSummary:
    cursor.execute(
        build_bmk_parser_sync_update_sql(
            target_relation_name,
            source_relation_name=source_relation_name,
        )
    )
    rows_updated = cursor_rowcount(cursor)
    cursor.execute(build_bmk_parser_sync_relation_summary_sql(target_relation_name))
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(
            "Bmk parser sync summary query returned no rows: "
            f"{target_relation_name}"
        )
    return BmkParserSyncRelationSummary(
        relation_name=target_relation_name,
        rows_updated=rows_updated,
        ambiguous_ob_rows=scalar_to_int(row[0], relation_name=target_relation_name),
        ob_rows=scalar_to_int(row[1], relation_name=target_relation_name),
        o_rows=scalar_to_int(row[2], relation_name=target_relation_name),
    )


def _refresh_downstream_summary_tables(cursor: DbapiCursorProtocol) -> None:
    for drop_sql, create_sql in (
        build_bmk_training_summary_refresh_sql(),
        build_bmk_quality_summary_refresh_sql(),
        build_bmk_unknown_summary_refresh_sql(),
    ):
        cursor.execute(drop_sql)
        cursor.execute(create_sql)


__all__ = ["sync_bmk_parser_fields_downstream"]
