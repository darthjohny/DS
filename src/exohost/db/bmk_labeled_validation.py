# Файл `bmk_labeled_validation.py` слоя `db`.
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

from exohost.db.bmk_labeled_contracts import (
    B_MK_EXTERNAL_LABELED_REQUIRED_CROSSMATCH_COLUMNS,
    B_MK_EXTERNAL_LABELED_REQUIRED_FILTERED_COLUMNS,
)
from exohost.db.relations import relation_columns


def validate_required_bmk_labeled_source_columns(
    engine: Engine,
    *,
    filtered_relation_name: str,
    crossmatch_relation_name: str,
) -> None:
    # Проверяем минимальный набор колонок в filtered и crossmatch relation перед materialization.
    filtered_columns = set(relation_columns(engine, filtered_relation_name))
    crossmatch_columns = set(relation_columns(engine, crossmatch_relation_name))
    missing_filtered_columns = [
        column_name
        for column_name in B_MK_EXTERNAL_LABELED_REQUIRED_FILTERED_COLUMNS
        if column_name not in filtered_columns
    ]
    missing_crossmatch_columns = [
        column_name
        for column_name in B_MK_EXTERNAL_LABELED_REQUIRED_CROSSMATCH_COLUMNS
        if column_name not in crossmatch_columns
    ]
    if missing_filtered_columns:
        raise ValueError(
            "Missing required filtered columns for B/mk external labeled materialization: "
            + ", ".join(missing_filtered_columns)
        )
    if missing_crossmatch_columns:
        raise ValueError(
            "Missing required crossmatch columns for B/mk external labeled materialization: "
            + ", ".join(missing_crossmatch_columns)
        )
