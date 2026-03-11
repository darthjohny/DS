"""Общие helpers для introspection relation в Postgres.

Модуль используется там, где нужно:

- разобрать полное имя relation на `schema.table`;
- проверить существование таблицы или view;
- получить упорядоченный список колонок из системного introspection API.
"""

from __future__ import annotations

import re

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def split_relation_name(
    relation_name: str,
    *,
    validate_identifiers: bool = False,
) -> tuple[str, str]:
    """Разбить имя relation на `schema` и `table`.

    Если схема не указана явно, используется `public`. При включённой
    валидации функция дополнительно проверяет, что оба идентификатора
    похожи на допустимые SQL-имена без произвольных символов.
    """
    parts = relation_name.split(".", 1)
    if len(parts) == 2:
        schema_name, table_name = parts
    else:
        schema_name, table_name = "public", relation_name

    if validate_identifiers:
        if not IDENTIFIER_PATTERN.match(schema_name):
            raise ValueError(f"Invalid schema name: {schema_name}")
        if not IDENTIFIER_PATTERN.match(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

    return schema_name, table_name


def relation_exists(
    engine: Engine,
    relation_name: str,
    *,
    include_views: bool = True,
    validate_identifiers: bool = False,
) -> bool:
    """Проверить существование таблицы или view в целевой схеме.

    Функция использует SQLAlchemy inspector и по умолчанию считает
    допустимыми как таблицы, так и view. Поведение можно ужесточить
    через `include_views=False`.
    """
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=validate_identifiers,
    )
    inspector = sa_inspect(engine)
    table_names = inspector.get_table_names(schema=schema_name)
    if table_name in table_names:
        return True
    if not include_views:
        return False
    return table_name in inspector.get_view_names(schema=schema_name)


def relation_columns(
    engine: Engine,
    relation_name: str,
    *,
    validate_identifiers: bool = False,
) -> tuple[str, ...]:
    """Вернуть упорядоченный список колонок таблицы или view.

    Источник данных
    ---------------
    Метаданные читаются через SQLAlchemy inspector из целевой схемы
    Postgres без запроса самих данных relation.
    """
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=validate_identifiers,
    )
    inspector = sa_inspect(engine)
    columns = inspector.get_columns(table_name, schema=schema_name)
    return tuple(str(item["name"]) for item in columns)
