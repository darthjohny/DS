# Файл `bmk_parser_sync_validation.py` слоя `db`.
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
    B_MK_PARSER_DERIVED_COLUMNS,
    B_MK_PARSER_SYNC_JOIN_COLUMNS,
    B_MK_PARSER_SYNC_SOURCE_RELATION_NAME,
    B_MK_PARSER_SYNC_TARGET_RELATION_NAMES,
)
from exohost.db.relations import relation_columns


def validate_bmk_parser_sync_columns(
    engine: Engine,
    *,
    source_relation_name: str = B_MK_PARSER_SYNC_SOURCE_RELATION_NAME,
) -> None:
    # Проверяем, что source и target relations содержат весь required column set.
    required_columns = B_MK_PARSER_SYNC_JOIN_COLUMNS + B_MK_PARSER_DERIVED_COLUMNS + (
        "external_catalog_name",
    )
    source_columns = set(
        relation_columns(engine, source_relation_name, validate_identifiers=True)
    )
    missing_source_columns = tuple(
        column_name
        for column_name in required_columns
        if column_name not in source_columns
    )
    if missing_source_columns:
        missing_columns_sql = ", ".join(missing_source_columns)
        raise RuntimeError(
            "Bmk parser sync source is missing required columns: "
            f"{missing_columns_sql}"
        )

    for target_relation_name in B_MK_PARSER_SYNC_TARGET_RELATION_NAMES:
        target_columns = set(
            relation_columns(engine, target_relation_name, validate_identifiers=True)
        )
        missing_target_columns = tuple(
            column_name
            for column_name in required_columns
            if column_name not in target_columns
        )
        if missing_target_columns:
            missing_columns_sql = ", ".join(missing_target_columns)
            raise RuntimeError(
                "Bmk parser sync target is missing required columns: "
                f"{target_relation_name} -> {missing_columns_sql}"
            )


__all__ = ["validate_bmk_parser_sync_columns"]
