# star_orchestrator.py
# ============================================================
# Назначение
# ------------------------------------------------------------
# Оркестратор пайплайна для проекта ВКР
# (Gaia DR3 + NASA hosts).
#
# Цель файла:
# связать два уже существующих слоя модели
# в единый воспроизводимый процесс:
#
# 1. Сначала физически распознать звезду
#    через Gaussian router.
# 2. Затем для ветки M/K/G/F dwarf
#    посчитать host-like similarity.
# 3. Потом применить дополнительные
#    факторы приоритизации.
# 4. При необходимости сохранить
#    результаты в БД.
#
# Что делает этот файл:
# - загружает Gaia-кандидатов;
# - загружает две модели с диска;
# - запускает router;
# - при необходимости пишет
#   router-результаты в БД;
# - разделяет поток на ветки;
# - считает host similarity
#   только для M/K/G/F dwarf;
# - формирует low-priority stub
#   для A/B/O и evolved;
# - собирает итоговую таблицу
#   приоритизации;
# - при необходимости пишет
#   итоговые результаты в БД.
#
# Важно:
# - здесь нет обучения моделей;
# - здесь нет EDA и графиков;
# - здесь нет подбора гиперпараметров;
# - здесь только orchestration,
#   decision layer
#   и запись результатов.
# ============================================================

"""Оркестратор пайплайна Router + Host Similarity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from uuid import uuid4

import pandas as pd
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

from gaussian_router import (
    FEATURES as ROUTER_FEATURES,
    RouterModel,
    load_router_model,
    make_engine_from_env,
    score_router_df,
)
from model_gaussian import load_model as load_host_model
from model_gaussian import score_df as score_host_df

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_INPUT_SOURCE = "public.gaia_dr3_training"
DEFAULT_ROUTER_RESULTS_TABLE = "lab.gaia_router_results"
DEFAULT_PRIORITY_RESULTS_TABLE = "lab.gaia_priority_results"

ROUTER_MODEL_PATH = DATA_DIR / "router_gaussian_params.json"
HOST_MODEL_PATH = DATA_DIR / "model_gaussian_params.json"
HOST_MODEL_VERSION = "gaussian_similarity_v1"

MKGF_CLASSES = {"M", "K", "G", "F"}

INPUT_COLUMNS: Tuple[str, ...] = (
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "bp_rp",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
    "mh_gspphot",
    "validation_factor",
)

ROUTER_RESULTS_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "source_id",
    "ra",
    "dec",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
    "predicted_spec_class",
    "predicted_evolution_stage",
    "router_label",
    "d_mahal_router",
    "router_similarity",
    "second_best_label",
    "margin",
    "router_model_version",
)

PRIORITY_RESULTS_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "source_id",
    "ra",
    "dec",
    "predicted_spec_class",
    "predicted_evolution_stage",
    "router_label",
    "d_mahal_router",
    "router_similarity",
    "gauss_label",
    "d_mahal",
    "similarity",
    "class_prior",
    "quality_factor",
    "metallicity_factor",
    "color_factor",
    "validation_factor",
    "final_score",
    "priority_tier",
    "reason_code",
    "router_model_version",
    "host_model_version",
)


@dataclass(frozen=True)
class PipelineRunResult:
    """Результат одного прогона orchestrator."""

    run_id: str
    router_results: pd.DataFrame
    priority_results: pd.DataFrame


def split_relation_name(relation_name: str) -> Tuple[str, str]:
    """Разделить имя relation на schema и relation."""
    parts = relation_name.split(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "public", relation_name


def relation_exists(engine: Engine, relation_name: str) -> bool:
    """Проверить, что таблица или view существует в БД."""
    schema, relation = split_relation_name(relation_name)
    inspector = sa_inspect(engine)
    return bool(
        relation in inspector.get_table_names(schema=schema)
        or relation in inspector.get_view_names(schema=schema)
    )


def relation_columns(engine: Engine, relation_name: str) -> List[str]:
    """Получить список колонок таблицы или view."""
    schema, relation = split_relation_name(relation_name)
    inspector = sa_inspect(engine)
    columns = inspector.get_columns(relation, schema=schema)
    return [str(item["name"]) for item in columns]


def make_run_id(prefix: str = "gaia_pipeline") -> str:
    """Собрать уникальный run_id с UTC-временем и коротким суффиксом."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"


def load_input_candidates(
    engine: Engine,
    source_name: str = DEFAULT_INPUT_SOURCE,
    limit: int | None = None,
) -> pd.DataFrame:
    """Загрузить Gaia-кандидатов с полным набором полей для пайплайна."""
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

    query = f"""
    SELECT
        {", ".join(selected)}
    FROM {schema}.{relation}
    WHERE source_id IS NOT NULL
      AND ra IS NOT NULL
      AND dec IS NOT NULL
      AND teff_gspphot IS NOT NULL
      AND logg_gspphot IS NOT NULL
      AND radius_gspphot IS NOT NULL
    {limit_sql};
    """
    df = pd.read_sql(query, engine)
    df = df.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    return ensure_decision_columns(df)


def ensure_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить недостающие decision-layer колонки с нейтральными значениями."""
    result = df.copy()
    defaults: Dict[str, float] = {
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


def load_models(
    router_model_path: Path = ROUTER_MODEL_PATH,
    host_model_path: Path = HOST_MODEL_PATH,
) -> Tuple[RouterModel, Dict[str, Any]]:
    """Загрузить router-модель и host-модель с диска."""
    if not router_model_path.exists():
        raise FileNotFoundError(
            f"Router model is missing: {router_model_path}"
        )
    if not host_model_path.exists():
        raise FileNotFoundError(f"Host model is missing: {host_model_path}")

    router_model = load_router_model(str(router_model_path))
    host_model = load_host_model(str(host_model_path))
    return router_model, host_model


def run_router(df: pd.DataFrame, router_model: RouterModel) -> pd.DataFrame:
    """Прогнать входной DataFrame через физический router."""
    scored = score_router_df(model=router_model, df=df)
    return scored.rename(columns={"model_version": "router_model_version"})


def save_router_results(
    df_router: pd.DataFrame,
    engine: Engine,
    table_name: str = DEFAULT_ROUTER_RESULTS_TABLE,
) -> None:
    """Сохранить router-результаты в lab.gaia_router_results."""
    if df_router.empty:
        return

    schema, table = split_relation_name(table_name)
    payload = df_router.loc[:, ROUTER_RESULTS_COLUMNS].copy()
    payload.to_sql(
        name=table,
        schema=schema,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )


def split_branches(df_router: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить поток на host-ветку и low-priority ветку."""
    host_mask = (
        df_router["predicted_spec_class"].isin(MKGF_CLASSES)
        & (df_router["predicted_evolution_stage"] == "dwarf")
    )
    df_host = df_router.loc[host_mask].copy()
    df_low = df_router.loc[~host_mask].copy()
    return df_host, df_low


def clip_unit_interval(value: float) -> float:
    """Ограничить численную величину диапазоном [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def class_prior(spec_class: Any) -> float:
    """Физический prior для задачи поиска перспективных rocky targets."""
    priors = {
        "M": 1.00,
        "K": 0.95,
        "G": 0.80,
        "F": 0.65,
        "A": 0.20,
        "B": 0.20,
        "O": 0.20,
    }
    return float(priors.get(str(spec_class), 0.20))


def ruwe_factor(value: Any) -> float:
    """Преобразовать RUWE в множитель качества астрометрии."""
    if pd.isna(value):
        return 0.85
    ruwe = float(value)
    if ruwe <= 1.10:
        return 1.00
    if ruwe <= 1.40:
        return 0.92
    if ruwe <= 2.00:
        return 0.70
    return 0.45


def parallax_precision_factor(value: Any) -> float:
    """Преобразовать parallax_over_error в множитель точности расстояния."""
    if pd.isna(value):
        return 0.75
    ratio = float(value)
    if ratio >= 20.0:
        return 1.00
    if ratio >= 10.0:
        return 0.92
    if ratio >= 5.0:
        return 0.78
    if ratio > 0.0:
        return 0.60
    return 0.40


def distance_factor(parallax: Any) -> float:
    """Использовать параллакс как мягкий proxy-фактор близости звезды."""
    if pd.isna(parallax):
        return 0.75
    plx = float(parallax)
    if plx >= 20.0:
        return 1.00
    if plx >= 10.0:
        return 0.92
    if plx >= 5.0:
        return 0.82
    if plx > 0.0:
        return 0.65
    return 0.45


def quality_factor(
    ruwe: Any,
    parallax_over_error: Any,
    parallax: Any,
) -> float:
    """Объединить качество астрометрии и расстояния в единый множитель."""
    values = (
        ruwe_factor(ruwe),
        parallax_precision_factor(parallax_over_error),
        distance_factor(parallax),
    )
    return clip_unit_interval(sum(values) / float(len(values)))


def metallicity_factor(value: Any) -> float:
    """Преобразовать [M/H] в осторожный множитель приоритета."""
    if pd.isna(value):
        return 1.00
    mh = float(value)
    if mh >= 0.20:
        return 1.00
    if mh >= -0.10:
        return 0.95
    if mh >= -0.40:
        return 0.85
    return 0.70


def color_factor(value: Any) -> float:
    """Использовать Gaia BP-RP как мягкое предпочтение более холодным звёздам."""
    if pd.isna(value):
        return 1.00
    bp_rp = float(value)
    if bp_rp >= 1.30:
        return 1.00
    if bp_rp >= 0.90:
        return 0.90
    if bp_rp >= 0.50:
        return 0.75
    return 0.60


def normalized_validation_factor(value: Any) -> float:
    """Сохранить validation_factor в диапазоне [0, 1]."""
    if pd.isna(value):
        return 1.00
    return clip_unit_interval(float(value))


def priority_tier_from_score(score: float) -> str:
    """Перевести непрерывный score в рабочий приоритетный tier."""
    if score >= 0.55:
        return "HIGH"
    if score >= 0.30:
        return "MEDIUM"
    return "LOW"


def stub_reason_code(spec_class: Any, evolution_stage: Any) -> str:
    """Объяснить, почему объект ушёл в ветку низкого приоритета."""
    spec = str(spec_class)
    stage = str(evolution_stage)
    if spec in {"A", "B", "O"}:
        return "HOT_STAR"
    if stage == "evolved":
        return "EVOLVED_STAR"
    return "FILTERED_OUT"


def iter_triplets(
    rows: Iterable[Tuple[Any, Any, Any]],
) -> Iterable[Tuple[Any, Any, Any]]:
    """Явный адаптер для трехзначных tuple-итераторов."""
    return rows


def apply_common_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить общие множители decision layer для обеих веток."""
    result = ensure_decision_columns(df)
    result["class_prior"] = [
        class_prior(spec_class)
        for spec_class in result["predicted_spec_class"]
    ]
    quality_rows = result[
        ["ruwe", "parallax_over_error", "parallax"]
    ].itertuples(index=False, name=None)
    result["quality_factor"] = [
        quality_factor(ruwe, plx_err, parallax)
        for ruwe, plx_err, parallax in iter_triplets(quality_rows)
    ]
    result["metallicity_factor"] = [
        metallicity_factor(value) for value in result["mh_gspphot"]
    ]
    result["color_factor"] = [
        color_factor(value) for value in result["bp_rp"]
    ]
    result["validation_factor"] = [
        normalized_validation_factor(value)
        for value in result["validation_factor"]
    ]
    return result


def host_model_version(host_model: Dict[str, Any]) -> str:
    """Собрать человекочитаемую версию host-модели."""
    meta = host_model.get("meta", {})
    shrink = meta.get("shrink_alpha")
    use_m_subclasses = meta.get("use_m_subclasses")
    if shrink is None or use_m_subclasses is None:
        return HOST_MODEL_VERSION
    return (
        f"{HOST_MODEL_VERSION}_"
        f"msub_{bool(use_m_subclasses)}_"
        f"shrink_{float(shrink):.2f}"
    )


def run_host_similarity(
    df_host: pd.DataFrame,
    host_model: Dict[str, Any],
) -> pd.DataFrame:
    """Посчитать host similarity только для физически допустимой MKGF-ветки."""
    if df_host.empty:
        return df_host.copy()

    scored = score_host_df(
        model=host_model,
        df=df_host,
        spec_class_col="predicted_spec_class",
    )
    scored = apply_common_factors(scored)

    scoring_rows = scored[
        [
            "similarity",
            "class_prior",
            "quality_factor",
            "metallicity_factor",
            "color_factor",
            "validation_factor",
        ]
    ].itertuples(index=False, name=None)
    scored["final_score"] = [
        clip_unit_interval(
            float(similarity)
            * float(class_prior_value)
            * float(quality_value)
            * float(metallicity_value)
            * float(color_value)
            * float(validation_value)
        )
        for (
            similarity,
            class_prior_value,
            quality_value,
            metallicity_value,
            color_value,
            validation_value,
        ) in scoring_rows
    ]
    scored["priority_tier"] = [
        priority_tier_from_score(float(score))
        for score in scored["final_score"]
    ]
    scored["reason_code"] = "HOST_SCORING"
    scored["host_model_version"] = host_model_version(host_model)
    return scored


def build_low_priority_stub(df_low: pd.DataFrame) -> pd.DataFrame:
    """Построить low-priority результат для A/B/O и evolved-звёзд."""
    if df_low.empty:
        return df_low.copy()

    result = apply_common_factors(df_low)
    result["gauss_label"] = None
    result["d_mahal"] = None
    result["similarity"] = None
    result["final_score"] = 0.0
    result["priority_tier"] = "LOW"
    result["reason_code"] = [
        stub_reason_code(spec_class, stage)
        for spec_class, stage in result[
            ["predicted_spec_class", "predicted_evolution_stage"]
        ].itertuples(index=False, name=None)
    ]
    result["host_model_version"] = None
    return result


def save_priority_results(
    df_priority: pd.DataFrame,
    engine: Engine,
    table_name: str = DEFAULT_PRIORITY_RESULTS_TABLE,
) -> None:
    """Сохранить итоговые результаты приоритизации в БД."""
    if df_priority.empty:
        return

    schema, table = split_relation_name(table_name)
    payload = df_priority.loc[:, PRIORITY_RESULTS_COLUMNS].copy()
    payload.to_sql(
        name=table,
        schema=schema,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )


def order_priority_results(df: pd.DataFrame) -> pd.DataFrame:
    """Отсортировать результаты в рабочем ranking-порядке."""
    return df.sort_values(
        by=["final_score", "router_similarity"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


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
    """Выполнить полный пайплайн Router + Host Similarity."""
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
    """Напечатать короткий preview одного безопасного прогона."""
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
    """Запустить безопасный preview пайплайна без записи в БД."""
    engine = make_engine_from_env()
    result = run_pipeline(
        engine=engine,
        input_source=DEFAULT_INPUT_SOURCE,
        limit=100,
        persist=False,
    )
    print_preview(result)


if __name__ == "__main__":
    main()
