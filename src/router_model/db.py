"""Bootstrap подключения и загрузка обучающей выборки для router.

Модуль инкапсулирует:

- создание engine для retrain-сценария router;
- загрузку reference-layer из Postgres;
- базовый контракт признаков, необходимых для физической классификации.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

from infra.db import make_engine_from_env as _make_engine_from_env

FEATURES: list[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
ROUTER_VIEW = "lab.v_gaia_router_training"


def make_engine_from_env() -> Engine:
    """Создать engine для загрузки router training source из Postgres.

    Функция использует общий bootstrap из `infra.db` и дополнительно
    запрещает очевидно шаблонные `DATABASE_URL`, чтобы retrain не
    стартовал на незаполненной конфигурации.
    """
    return _make_engine_from_env(
        reject_placeholder_url=True,
    )


def load_router_training_from_db(
    engine: Engine,
    view_name: str = ROUTER_VIEW,
) -> pd.DataFrame:
    """Загрузить reference-layer для обучения router из Postgres.

    Источник данных
    ---------------
    По умолчанию читается relation `lab.v_gaia_router_training`.
    Из источника выбираются только строки с поддерживаемыми значениями
    `spec_class` и `evolution_stage`, а также базовые признаки из
    `FEATURES`.

    Исключения
    ----------
    RuntimeError
        Если указанная relation не существует в БД.
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
            f"Router training source does not exist: {view_name}"
        )

    query = f"""
    SELECT
        source_id,
        spec_class,
        evolution_stage,
        {", ".join(FEATURES)}
    FROM {schema}.{rel}
    WHERE spec_class IN ('A','B','F','G','K','M','O')
      AND evolution_stage IN ('dwarf','evolved');
    """
    return pd.read_sql(query, engine)


__all__ = [
    "FEATURES",
    "ROUTER_VIEW",
    "load_router_training_from_db",
    "make_engine_from_env",
]
