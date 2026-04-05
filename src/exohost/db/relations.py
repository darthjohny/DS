# Файл `relations.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import re

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(identifier: str) -> str:
    # Проверяем безопасный SQL identifier и возвращаем его же для fluent-style вызовов.
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"Invalid identifier: {identifier}")
    return identifier


def split_relation_name(
    relation_name: str,
    *,
    validate_identifiers: bool = False,
) -> tuple[str, str]:
    # Разбиваем relation на schema.table и при необходимости валидируем.
    parts = relation_name.split(".", 1)
    if len(parts) == 2:
        schema_name, table_name = parts
    else:
        schema_name, table_name = "public", relation_name

    if validate_identifiers:
        if not IDENTIFIER_PATTERN.fullmatch(schema_name):
            raise ValueError(f"Invalid schema name: {schema_name}")
        if not IDENTIFIER_PATTERN.fullmatch(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

    return schema_name, table_name


def quote_identifier(identifier: str) -> str:
    # Безопасно заключаем identifier в двойные кавычки после валидации.
    validate_identifier(identifier)
    return f'"{identifier}"'


def quote_relation_name(
    relation_name: str,
    *,
    validate_identifiers: bool = False,
) -> str:
    # Приводим schema.table к безопасному SQL-формату с кавычками.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=validate_identifiers,
    )
    return f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"


def relation_exists(
    engine: Engine,
    relation_name: str,
    *,
    include_views: bool = True,
    validate_identifiers: bool = False,
) -> bool:
    # Проверяем существование таблицы или view через SQLAlchemy inspector.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=validate_identifiers,
    )
    inspector = sa_inspect(engine)

    if table_name in inspector.get_table_names(schema=schema_name):
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
    # Возвращаем колонки relation в порядке, который дает introspection API.
    schema_name, table_name = split_relation_name(
        relation_name,
        validate_identifiers=validate_identifiers,
    )
    inspector = sa_inspect(engine)
    columns = inspector.get_columns(table_name, schema=schema_name)
    return tuple(str(item["name"]) for item in columns)
