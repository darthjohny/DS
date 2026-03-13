"""DB-helpers загрузки данных для обучения host-модели.

Модуль отвечает за чтение train relations из Postgres для двух сценариев:

- legacy Gaussian model по host-популяции;
- contrastive `host-vs-field` обучение из явного view или из стандартных
  project sources.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

from host_model.constants import (
    CONTRASTIVE_POPULATION_COLUMN,
    CONTRASTIVE_VIEW_ENV,
    DEFAULT_CONTRASTIVE_FIELD_VIEW,
    DEFAULT_CONTRASTIVE_HOST_VIEW,
    DWARF_CLASSES,
    FEATURES,
    LOGG_DWARF_MIN,
)
from host_model.training_data import prepare_contrastive_training_df
from infra.db import make_engine_from_env as _make_engine_from_env


def make_engine_from_env() -> Engine:
    """Создать engine для retrain host-модели из параметров окружения."""
    return _make_engine_from_env(
        reject_placeholder_url=True,
    )


def load_dwarfs_from_db(
    engine: Engine,
    view_name: str = "lab.v_nasa_gaia_train_dwarfs",
) -> pd.DataFrame:
    """Загрузить MKGF dwarfs для legacy host-модели.

    Источник данных
    ---------------
    Если переданный `view_name` существует, используется он. Иначе
    функция откатывается к `lab.v_nasa_gaia_train_classified` и сама
    применяет фильтр `logg_gspphot >= LOGG_DWARF_MIN`.
    """
    if "." in view_name:
        schema, rel = view_name.split(".", 1)
    else:
        schema, rel = "public", view_name

    inspector = sa_inspect(engine)
    has_rel = (
        rel in inspector.get_table_names(schema=schema)
        or rel in inspector.get_view_names(schema=schema)
    )

    if has_rel:
        source = f"{schema}.{rel}"
        query = f"""
        SELECT spec_class, {", ".join(FEATURES)}
        FROM {source}
        WHERE spec_class IN ('M','K','G','F');
        """
    else:
        source = "lab.v_nasa_gaia_train_classified"
        query = f"""
        SELECT spec_class, {", ".join(FEATURES)}
        FROM {source}
        WHERE spec_class IN ('M','K','G','F')
          AND logg_gspphot >= {LOGG_DWARF_MIN};
        """

    return pd.read_sql(query, engine)


def load_contrastive_training_from_db(
    engine: Engine,
    view_name: str,
    population_col: str = CONTRASTIVE_POPULATION_COLUMN,
) -> pd.DataFrame:
    """Загрузить и провалидировать contrastive source из Postgres.

    Источник должен содержать `spec_class`, бинарную колонку популяции
    и признаки из `FEATURES`. После чтения данные дополнительно проходят
    проверку через `prepare_contrastive_training_df()`.
    """
    if "." in view_name:
        schema, rel = view_name.split(".", 1)
    else:
        schema, rel = "public", view_name

    inspector = sa_inspect(engine)
    has_rel = (
        rel in inspector.get_table_names(schema=schema)
        or rel in inspector.get_view_names(schema=schema)
    )
    if not has_rel:
        raise RuntimeError(
            f"Contrastive training source does not exist: {view_name}"
        )

    available_columns = {
        str(column["name"])
        for column in inspector.get_columns(rel, schema=schema)
    }
    required_columns = {"spec_class", population_col, *FEATURES}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(
            "Contrastive training source is missing required DB columns: "
            f"{', '.join(missing_columns)}"
        )

    query = f"""
    SELECT
        spec_class,
        {population_col},
        {", ".join(FEATURES)}
    FROM {schema}.{rel}
    WHERE spec_class IN {tuple(DWARF_CLASSES)};
    """
    raw_df = pd.read_sql(query, engine)
    return prepare_contrastive_training_df(
        raw_df,
        population_col=population_col,
    )


def load_default_contrastive_training_from_db(
    engine: Engine,
    population_col: str = CONTRASTIVE_POPULATION_COLUMN,
    host_view: str = DEFAULT_CONTRASTIVE_HOST_VIEW,
    field_view: str = DEFAULT_CONTRASTIVE_FIELD_VIEW,
) -> pd.DataFrame:
    """Собрать contrastive train set из стандартных project relations.

    Host-популяция берётся из `DEFAULT_CONTRASTIVE_HOST_VIEW`, field-
    популяция — из `DEFAULT_CONTRASTIVE_FIELD_VIEW`, после чего наборы
    объединяются и валидируются как единый contrastive dataset.
    """
    host_df = load_dwarfs_from_db(engine, view_name=host_view).copy()
    host_df[population_col] = True

    if "." in field_view:
        schema, rel = field_view.split(".", 1)
    else:
        schema, rel = "public", field_view
    inspector = sa_inspect(engine)
    has_rel = (
        rel in inspector.get_table_names(schema=schema)
        or rel in inspector.get_view_names(schema=schema)
    )
    if not has_rel:
        raise RuntimeError(
            f"Default contrastive field source does not exist: {field_view}"
        )

    available_columns = {
        str(column["name"])
        for column in inspector.get_columns(rel, schema=schema)
    }
    required_columns = {"spec_class", *FEATURES}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(
            "Default contrastive field source is missing required DB columns: "
            f"{', '.join(missing_columns)}"
        )

    query = f"""
    SELECT
        spec_class,
        {", ".join(FEATURES)}
    FROM {schema}.{rel}
    WHERE spec_class IN {tuple(DWARF_CLASSES)};
    """
    field_only_df = pd.read_sql(query, engine)
    field_only_df[population_col] = False

    combined = pd.concat(
        [
            host_df[["spec_class", population_col, *FEATURES]],
            field_only_df[["spec_class", population_col, *FEATURES]],
        ],
        ignore_index=True,
        sort=False,
    )
    return prepare_contrastive_training_df(
        combined,
        population_col=population_col,
    )


def resolve_contrastive_view_name(view_name: str | None) -> str:
    """Определить имя DB view для contrastive retraining.

    Приоритет такой:
    1. явный аргумент CLI;
    2. переменная окружения `CONTRASTIVE_HOST_FIELD_VIEW`;
    3. пустая строка, означающая использование стандартных sources.
    """
    if view_name:
        return str(view_name)
    env_value = os.getenv(CONTRASTIVE_VIEW_ENV, "").strip()
    if env_value:
        return env_value
    return ""


__all__ = [
    "load_contrastive_training_from_db",
    "load_default_contrastive_training_from_db",
    "load_dwarfs_from_db",
    "make_engine_from_env",
    "resolve_contrastive_view_name",
]
