"""Вспомогательные функции для DB relation в боевом pipeline."""

from __future__ import annotations

from sqlalchemy.engine import Engine

from infra.relations import relation_columns as _relation_columns
from infra.relations import relation_exists as _relation_exists
from infra.relations import split_relation_name as _split_relation_name


def split_relation_name(relation_name: str) -> tuple[str, str]:
    """Разбить имя relation на схему и имя таблицы или view."""
    return _split_relation_name(relation_name)


def relation_exists(engine: Engine, relation_name: str) -> bool:
    """Проверить, что таблица или view существует в БД."""
    return _relation_exists(
        engine,
        relation_name,
        include_views=True,
    )


def relation_columns(engine: Engine, relation_name: str) -> list[str]:
    """Получить список колонок relation для записи и интроспекции."""
    return list(_relation_columns(engine, relation_name))


__all__ = ["relation_columns", "relation_exists", "split_relation_name"]
