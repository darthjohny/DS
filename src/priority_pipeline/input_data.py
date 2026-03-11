"""Загрузка входных данных и моделей для боевого pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy.engine import Engine

from gaussian_router import (
    FEATURES as ROUTER_FEATURES,
)
from gaussian_router import (
    RouterModel,
    load_router_model,
)
from host_model import ContrastiveGaussianModel, load_contrastive_model
from priority_pipeline.constants import (
    DEFAULT_INPUT_SOURCE,
    HOST_MODEL_PATH,
    INPUT_COLUMNS,
    ROUTER_MODEL_PATH,
)
from priority_pipeline.relations import (
    relation_columns,
    relation_exists,
    split_relation_name,
)


def make_run_id(prefix: str = "gaia_pipeline") -> str:
    """Собрать уникальный `run_id` из UTC-времени и короткого суффикса."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"


def ensure_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить отсутствующие decision-layer колонки с нейтральными значениями.

    Функция готовит входной DataFrame к общему контракту боевого
    pipeline, даже если часть soft-factor колонок отсутствует в исходной
    relation.
    """
    result = df.copy()
    defaults: dict[str, float] = {
        "parallax": float("nan"),
        "parallax_over_error": float("nan"),
        "ruwe": float("nan"),
        "bp_rp": float("nan"),
        "mh_gspphot": float("nan"),
        "validation_factor": 1.0,
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    return result


def load_input_candidates(
    engine: Engine,
    source_name: str = DEFAULT_INPUT_SOURCE,
    limit: int | None = None,
) -> pd.DataFrame:
    """Загрузить входных кандидатов Gaia для боевого pipeline.

    Источник данных
    ---------------
    Читает relation из Postgres, проверяет наличие обязательных колонок
    для router и затем выбирает одну детерминированную строку на
    `source_id`, используя quality-aware порядок сортировки.

    Возвращает
    ----------
    pd.DataFrame
        Нормализованный входной DataFrame с колонками для router и
        decision layer.
    """
    if not relation_exists(engine, source_name):
        raise RuntimeError(f"Input source does not exist: {source_name}")

    available = set(relation_columns(engine, source_name))
    required = ("source_id", "ra", "dec", *ROUTER_FEATURES)
    missing = [column for column in required if column not in available]
    if missing:
        raise RuntimeError(
            "Input source is missing required columns: "
            f"{', '.join(missing)}"
        )

    selected = [column for column in INPUT_COLUMNS if column in available]
    schema, relation = split_relation_name(source_name)
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""

    quality_order: list[str] = []
    if "parallax_over_error" in available:
        quality_order.append("base.parallax_over_error DESC NULLS LAST")
    if "ruwe" in available:
        quality_order.append("base.ruwe ASC NULLS LAST")
    if "validation_factor" in available:
        quality_order.append("base.validation_factor DESC NULLS LAST")

    quality_order.extend(
        [
            "base.source_id ASC",
            "base.ra ASC",
            "base.dec ASC",
            "base.teff_gspphot ASC",
            "base.logg_gspphot ASC",
            "base.radius_gspphot ASC",
        ]
    )
    order_sql = ",\n                ".join(quality_order)

    query = f"""
    WITH ranked AS (
        SELECT
            {", ".join(f"base.{column}" for column in selected)},
            ROW_NUMBER() OVER (
                PARTITION BY base.source_id
                ORDER BY {order_sql}
            ) AS rn
        FROM {schema}.{relation} AS base
        WHERE base.source_id IS NOT NULL
          AND base.ra IS NOT NULL
          AND base.dec IS NOT NULL
          AND base.teff_gspphot IS NOT NULL
          AND base.logg_gspphot IS NOT NULL
          AND base.radius_gspphot IS NOT NULL
    )
    SELECT
        {", ".join(selected)}
    FROM ranked
    WHERE rn = 1
    ORDER BY source_id ASC
    {limit_sql};
    """
    df = pd.read_sql(query, engine)
    return ensure_decision_columns(df.reset_index(drop=True))


def load_models(
    router_model_path: Path = ROUTER_MODEL_PATH,
    host_model_path: Path = HOST_MODEL_PATH,
) -> tuple[RouterModel, ContrastiveGaussianModel]:
    """Загрузить router и контрастивную host-модель с диска.

    Функция читает боевые артефакты из `data/` и проверяет, что
    host-модель соответствует текущему contrastive runtime-контракту.
    """
    if not router_model_path.exists():
        raise FileNotFoundError(
            f"Router model is missing: {router_model_path}"
        )
    if not host_model_path.exists():
        raise FileNotFoundError(f"Host model is missing: {host_model_path}")

    router_model = load_router_model(str(router_model_path))
    host_model = load_contrastive_model(str(host_model_path))
    return router_model, host_model


__all__ = [
    "ensure_decision_columns",
    "load_input_candidates",
    "load_models",
    "make_run_id",
]
