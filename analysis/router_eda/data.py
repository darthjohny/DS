"""Загрузка и подготовка данных для router EDA."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from analysis.router_eda.constants import FEATURES, ROUTER_VIEW
from infra.db import make_engine_from_env as _make_engine_from_env


def make_engine_from_env() -> Engine:
    """Создать SQLAlchemy engine из `DATABASE_URL` или `PG*` переменных."""
    return _make_engine_from_env(
        reject_placeholder_url=True,
    )


def read_sql_frame(engine: Engine, query: str) -> pd.DataFrame:
    """Типизированная обёртка над `pandas.read_sql` для локального EDA."""
    return pd.read_sql(query, engine)


def feature_frame(df_part: pd.DataFrame) -> pd.DataFrame:
    """Вернуть подтаблицу с базовыми router-признаками."""
    return df_part[FEATURES]


def make_router_label(spec_class: Any, evolution_stage: Any) -> str:
    """Собрать стабильную router-метку из класса и стадии."""
    spec_part = str(spec_class).strip().upper()
    stage_part = str(evolution_stage).strip().lower()
    return f"{spec_part}_{stage_part}"


def ensure_router_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить колонку `router_label`, если её ещё нет."""
    result = df.copy()
    result["router_label"] = [
        make_router_label(
            spec_class=spec_class,
            evolution_stage=evolution_stage,
        )
        for spec_class, evolution_stage in zip(
            result["spec_class"],
            result["evolution_stage"],
            strict=True,
        )
    ]
    return result


def build_class_counts(df_router: pd.DataFrame) -> pd.DataFrame:
    """Собрать агрегированные количества по классу и стадии эволюции."""
    counts = (
        df_router.groupby(
            ["spec_class", "evolution_stage", "router_label"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_objects"})
        .sort_values(
            ["spec_class", "evolution_stage", "router_label"],
            ignore_index=True,
        )
    )
    return counts


def build_router_training_query() -> str:
    """Собрать SQL-запрос для router reference view."""
    return f"""
    SELECT
        source_id,
        spec_class,
        evolution_stage,
        {", ".join(FEATURES)}
    FROM {ROUTER_VIEW}
    WHERE spec_class IN ('A','B','F','G','K','M','O')
      AND evolution_stage IN ('dwarf','evolved');
    """
