"""Функции записи результатов боевого pipeline в БД.

Модуль согласует результат рантайма со схемой целевых таблиц и сохраняет
только те колонки, которые одновременно:

- есть в DataFrame результата;
- разрешены контрактом `*_RESULTS_COLUMNS`;
- реально существуют в целевой relation БД.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from priority_pipeline.constants import (
    DEFAULT_PRIORITY_RESULTS_TABLE,
    DEFAULT_ROUTER_RESULTS_TABLE,
    PRIORITY_REQUIRED_DB_COLUMNS,
    PRIORITY_RESULTS_COLUMNS,
    ROUTER_REQUIRED_DB_COLUMNS,
    ROUTER_RESULTS_COLUMNS,
)
from priority_pipeline.relations import relation_columns, split_relation_name


def build_persist_payload(
    df: pd.DataFrame,
    ordered_columns: tuple[str, ...],
    available_columns: list[str],
    required_columns: tuple[str, ...],
    table_name: str,
) -> pd.DataFrame:
    """Собрать совместимый с БД payload для записи результата.

    Функция проверяет, что обязательные колонки присутствуют и в целевой
    relation, и в DataFrame результата, после чего отбирает только
    совместимое подмножество столбцов в каноническом порядке.
    """
    available_set = set(available_columns)
    missing_required_in_relation = [
        column for column in required_columns if column not in available_set
    ]
    if missing_required_in_relation:
        raise RuntimeError(
            f"Result table {table_name} is missing required columns: "
            f"{', '.join(missing_required_in_relation)}"
        )

    missing_required_in_df = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_required_in_df:
        raise RuntimeError(
            "Pipeline result frame is missing required persist columns: "
            f"{', '.join(missing_required_in_df)}"
        )

    writable_columns = [
        column
        for column in ordered_columns
        if column in df.columns and column in available_set
    ]
    if not writable_columns:
        raise RuntimeError(
            f"No compatible columns remain for DB persist into {table_name}."
        )
    return df.loc[:, writable_columns].copy()


def save_router_results(
    df_router: pd.DataFrame,
    engine: Engine,
    table_name: str = DEFAULT_ROUTER_RESULTS_TABLE,
) -> None:
    """Сохранить router-результаты в целевую relation БД.

    Побочные эффекты
    ----------------
    Выполняет `to_sql(..., if_exists="append")` в таблицу router-результатов.
    """
    if df_router.empty:
        return

    schema, table = split_relation_name(table_name)
    available_columns = relation_columns(engine, table_name)
    payload = build_persist_payload(
        df=df_router,
        ordered_columns=ROUTER_RESULTS_COLUMNS,
        available_columns=available_columns,
        required_columns=ROUTER_REQUIRED_DB_COLUMNS,
        table_name=table_name,
    )
    payload.to_sql(
        name=table,
        schema=schema,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )


def save_priority_results(
    df_priority: pd.DataFrame,
    engine: Engine,
    table_name: str = DEFAULT_PRIORITY_RESULTS_TABLE,
) -> None:
    """Сохранить итоговые результаты ранжирования в целевую relation БД.

    Побочные эффекты
    ----------------
    Выполняет `to_sql(..., if_exists="append")` в таблицу итоговой приоритизации.
    """
    if df_priority.empty:
        return

    schema, table = split_relation_name(table_name)
    available_columns = relation_columns(engine, table_name)
    payload = build_persist_payload(
        df=df_priority,
        ordered_columns=PRIORITY_RESULTS_COLUMNS,
        available_columns=available_columns,
        required_columns=PRIORITY_REQUIRED_DB_COLUMNS,
        table_name=table_name,
    )
    payload.to_sql(
        name=table,
        schema=schema,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )


__all__ = [
    "build_persist_payload",
    "save_priority_results",
    "save_router_results",
]
