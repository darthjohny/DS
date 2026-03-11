"""Загрузка данных и базовый scoring для офлайн-калибровки."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from host_model import score_df_contrastive as score_host_df
from input_layer import REGISTRY_TABLE
from priority_pipeline import (
    load_input_candidates,
    run_router,
    split_branches,
)


@dataclass(frozen=True)
class ReadyDatasetRecord:
    """Последняя registry-запись для входного датасета со статусом `READY`."""

    relation_name: str
    source_name: str
    status: str
    row_count: int
    validated_at: datetime | None


@dataclass(frozen=True)
class BaseScoringResult:
    """Результат базового preview-прогона до калиброванного `decision layer`.

    Хранит входной batch, router output, разбиение на ветки и результат
    host-scoring до применения офлайн-калибровочных факторов.
    """

    input_df: pd.DataFrame
    router_df: pd.DataFrame
    host_input_df: pd.DataFrame
    low_input_df: pd.DataFrame
    host_scored_df: pd.DataFrame


def make_run_id(prefix: str = "decision_calibration") -> str:
    """Собрать уникальный `run_id` для одной калибровочной итерации."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def fetch_ready_dataset_record(
    engine: Engine,
    relation_name: str,
) -> ReadyDatasetRecord:
    """Получить последнюю registry-запись и потребовать статус `READY`.

    Источник данных
    ---------------
    Читает `input_layer.REGISTRY_TABLE` и выбирает самую свежую запись
    по `validated_at` для заданного relation.
    """
    query = text(
        f"""
        SELECT
            relation_name,
            source_name,
            status,
            row_count,
            validated_at
        FROM {REGISTRY_TABLE}
        WHERE relation_name = :relation_name
        ORDER BY validated_at DESC
        LIMIT 1;
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            query,
            {"relation_name": relation_name},
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "Dataset is missing in registry. "
            "Validate it with input_layer.py first."
        )

    record = ReadyDatasetRecord(
        relation_name=str(row["relation_name"]),
        source_name=str(row["source_name"]),
        status=str(row["status"]),
        row_count=int(row["row_count"]),
        validated_at=row["validated_at"],
    )
    if record.status != "READY":
        raise RuntimeError(
            f"Dataset status must be READY, got {record.status}."
        )
    return record


def load_ready_input_dataset(
    engine: Engine,
    relation_name: str,
    limit: int | None,
) -> tuple[ReadyDatasetRecord, pd.DataFrame]:
    """Загрузить только тот relation, который уже прошёл `READY` validation."""
    record = fetch_ready_dataset_record(engine, relation_name)
    df = load_input_candidates(
        engine=engine,
        source_name=relation_name,
        limit=limit,
    )
    return record, df


def run_base_scoring(
    df_input: pd.DataFrame,
    router_model: Any,
    host_model: Mapping[str, Any],
) -> BaseScoringResult:
    """Выполнить router и host-scoring до применения калибровочной формулы.

    Функция переиспользует production helpers из `priority_pipeline`,
    но останавливается на стадии base scoring, не рассчитывая итоговый
    offline `final_score`.
    """
    router_df = run_router(df_input, router_model)
    host_input_df, low_input_df = split_branches(router_df)

    if host_input_df.empty:
        host_scored_df = host_input_df.copy()
    else:
        host_scored_df = score_host_df(
            model=dict(host_model),
            df=host_input_df,
            spec_class_col="predicted_spec_class",
        )

    return BaseScoringResult(
        input_df=df_input,
        router_df=router_df,
        host_input_df=host_input_df,
        low_input_df=low_input_df,
        host_scored_df=host_scored_df,
    )


__all__ = [
    "BaseScoringResult",
    "ReadyDatasetRecord",
    "fetch_ready_dataset_record",
    "load_ready_input_dataset",
    "make_run_id",
    "run_base_scoring",
]
