"""Загрузка и подготовка данных для host EDA."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sqlalchemy.engine import Engine

from analysis.host_eda.constants import (
    FEATURES,
    QUERY_ABO_REF,
    QUERY_ALL_MKGF,
    QUERY_DWARFS_MKGF,
    QUERY_EVOLVED_MKGF,
)
from infra.db import make_engine_from_env as _make_engine_from_env


def make_engine_from_env() -> Engine:
    """Создать SQLAlchemy engine для локального доступа EDA к БД."""
    return _make_engine_from_env(
        missing_message=(
            "Параметры подключения к БД не найдены. "
            "Задай DATABASE_URL или набор PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."
        ),
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Создать engine лениво, чтобы import модуля не ходил в БД."""
    return make_engine_from_env()


def read_sql_frame(query: str) -> pd.DataFrame:
    """Типизированная обёртка над `pandas.read_sql` для host EDA."""
    return pd.read_sql(query, get_engine())


def feature_frame(df_part: pd.DataFrame) -> pd.DataFrame:
    """Вернуть подтаблицу с базовыми признаками для анализа."""
    return df_part[FEATURES]


def load_all_mkgf() -> pd.DataFrame:
    """Загрузить базовую MKGF training view."""
    return read_sql_frame(QUERY_ALL_MKGF)


def load_dwarfs_mkgf() -> pd.DataFrame:
    """Загрузить MKGF view только для dwarf-объектов."""
    return read_sql_frame(QUERY_DWARFS_MKGF)


def load_evolved_mkgf() -> pd.DataFrame:
    """Загрузить MKGF view только для evolved-объектов."""
    return read_sql_frame(QUERY_EVOLVED_MKGF)


def load_abo_ref() -> pd.DataFrame:
    """Загрузить reference-layer для A/B/O объектов."""
    return read_sql_frame(QUERY_ABO_REF)
