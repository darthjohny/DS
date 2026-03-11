"""Основной сценарий выполнения боевого pipeline приоритизации."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from gaussian_router import RouterModel, make_engine_from_env, score_router_df
from host_model import ContrastiveGaussianModel
from priority_pipeline.constants import (
    DEFAULT_INPUT_SOURCE,
    DEFAULT_PRIORITY_RESULTS_TABLE,
    DEFAULT_ROUTER_RESULTS_TABLE,
    HOST_MODEL_PATH,
    PRIORITY_RESULTS_COLUMNS,
    ROUTER_MODEL_PATH,
)
from priority_pipeline.contracts import PipelineRunResult
from priority_pipeline.decision import (
    build_low_priority_stub,
    order_priority_results,
    run_host_similarity,
    split_branches,
)
from priority_pipeline.input_data import (
    load_input_candidates,
    load_models,
    make_run_id,
)
from priority_pipeline.persist import save_priority_results, save_router_results


def run_router(df: pd.DataFrame, router_model: RouterModel) -> pd.DataFrame:
    """Запустить физический router на входном DataFrame.

    Функция оборачивает `score_router_df()` и приводит имя поля версии
    модели к runtime-контракту `router_model_version`.
    """
    scored = score_router_df(model=router_model, df=df)
    return scored.rename(columns={"model_version": "router_model_version"})


def run_pipeline(
    engine: Engine,
    input_source: str = DEFAULT_INPUT_SOURCE,
    limit: int | None = None,
    persist: bool = False,
    router_results_table: str = DEFAULT_ROUTER_RESULTS_TABLE,
    priority_results_table: str = DEFAULT_PRIORITY_RESULTS_TABLE,
    router_model_path: Path = ROUTER_MODEL_PATH,
    host_model_path: Path = HOST_MODEL_PATH,
) -> PipelineRunResult:
    """Выполнить полный боевой pipeline end-to-end.

    Порядок выполнения
    ------------------
    1. Загрузить входной batch из Postgres.
    2. Загрузить router и host artifacts с диска.
    3. Выполнить router scoring.
    4. Разделить поток на host-ветку и low-priority ветку.
    5. Применить contrastive host-scoring к MKGF dwarf.
    6. Собрать итоговый ranking DataFrame.
    7. При `persist=True` записать router и priority результаты в БД.
    """
    df_input = load_input_candidates(
        engine=engine,
        source_name=input_source,
        limit=limit,
    )
    router_model, host_model = load_models(
        router_model_path=router_model_path,
        host_model_path=host_model_path,
    )

    run_id = make_run_id()
    df_router = run_router(df=df_input, router_model=router_model)
    df_router["run_id"] = run_id

    if persist:
        save_router_results(
            df_router=df_router,
            engine=engine,
            table_name=router_results_table,
        )

    df_host, df_low = split_branches(df_router)
    host_results = run_host_similarity(df_host=df_host, host_model=host_model)
    low_results = build_low_priority_stub(df_low=df_low)

    parts = [frame for frame in (host_results, low_results) if not frame.empty]
    if parts:
        df_priority = pd.concat(parts, ignore_index=True, sort=False)
        df_priority = order_priority_results(df_priority)
    else:
        df_priority = pd.DataFrame(columns=PRIORITY_RESULTS_COLUMNS)

    if persist:
        save_priority_results(
            df_priority=df_priority,
            engine=engine,
            table_name=priority_results_table,
        )

    return PipelineRunResult(
        run_id=run_id,
        router_results=df_router,
        priority_results=df_priority,
    )


def print_preview(result: PipelineRunResult) -> None:
    """Вывести краткий preview безопасного прогона pipeline в stdout."""
    print(f"run_id={result.run_id}")
    print(f"router_rows={len(result.router_results)}")
    print(f"priority_rows={len(result.priority_results)}")

    if not result.priority_results.empty:
        preview_columns = [
            "source_id",
            "predicted_spec_class",
            "predicted_evolution_stage",
            "router_label",
            "gauss_label",
            "final_score",
            "priority_tier",
            "reason_code",
        ]
        print(
            result.priority_results[preview_columns]
            .head(10)
            .to_string(index=False)
        )


def main() -> None:
    """Запустить безопасный preview боевого pipeline без записи в БД."""
    engine = make_engine_from_env()
    result = run_pipeline(
        engine=engine,
        input_source=DEFAULT_INPUT_SOURCE,
        limit=100,
        persist=False,
    )
    print_preview(result)


__all__ = [
    "ContrastiveGaussianModel",
    "PipelineRunResult",
    "run_pipeline",
    "run_router",
    "print_preview",
    "main",
]
