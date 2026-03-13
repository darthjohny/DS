"""Загрузка benchmark dataset и детерминированный split для comparison-layer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

from analysis.model_comparison.contracts import (
    DEFAULT_BENCHMARK_SOURCES,
    DEFAULT_COMPARISON_PROTOCOL,
    DEFAULT_SPLIT_CONFIG,
    BenchmarkSources,
    ComparisonProtocol,
    SplitConfig,
)
from analysis.model_comparison.tuning import (
    build_stratify_labels,
    validate_cross_validation_inputs,
)
from host_model import normalize_host_flag
from infra.db import make_engine_from_env as _make_engine_from_env


@dataclass(slots=True)
class BenchmarkSplit:
    """Разбиение benchmark dataset на full/train/test части."""

    full_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def make_engine_from_env() -> Engine:
    """Создать SQLAlchemy engine для comparison-layer."""
    return _make_engine_from_env(
        reject_placeholder_url=True,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Создать engine лениво, чтобы import comparison-layer не ходил в БД."""
    return make_engine_from_env()


def parse_relation_name(relation_name: str) -> tuple[str, str]:
    """Разобрать relation name на schema и relation."""
    if "." in relation_name:
        schema, relation = relation_name.split(".", 1)
        return schema, relation
    return "public", relation_name


def relation_exists(engine: Engine, relation_name: str) -> bool:
    """Проверить существование relation или view в Postgres."""
    schema, relation = parse_relation_name(relation_name)
    inspector = sa_inspect(engine)
    return (
        relation in inspector.get_table_names(schema=schema)
        or relation in inspector.get_view_names(schema=schema)
    )


def read_sql_frame(engine: Engine, query: str) -> pd.DataFrame:
    """Типизированная обёртка над `pandas.read_sql` для comparison-layer."""
    return pd.read_sql(query, engine)


def build_population_query(
    relation_name: str,
    *,
    is_host: bool,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> str:
    """Собрать SQL-запрос для одной benchmark-популяции."""
    schema, relation = parse_relation_name(relation_name)
    population_sql = "TRUE" if is_host else "FALSE"
    allowed_classes = ", ".join(f"'{spec_class}'" for spec_class in sources.allowed_classes)
    return f"""
    SELECT
        {sources.source_id_col},
        {sources.class_col},
        {", ".join(sources.feature_columns)},
        {population_sql} AS {sources.population_col}
    FROM {schema}.{relation}
    WHERE {sources.class_col} IN ({allowed_classes});
    """


def prepare_benchmark_dataset(
    df_benchmark: pd.DataFrame,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Провалидировать и нормализовать benchmark dataset.

    Функция:

    - оставляет только канонический набор колонок;
    - нормализует `spec_class` и `is_host`;
    - удаляет строки с пропусками в обязательных полях;
    - запрещает повторяющиеся `source_id`, чтобы избежать leakage.
    """
    required_columns = [
        sources.source_id_col,
        sources.class_col,
        sources.population_col,
        *sources.feature_columns,
    ]
    missing_columns = [
        column for column in required_columns if column not in df_benchmark.columns
    ]
    if missing_columns:
        raise ValueError(
            "Benchmark dataset is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    result = df_benchmark.loc[:, required_columns].copy()
    result[sources.class_col] = (
        result[sources.class_col].astype(str).str.strip().str.upper()
    )
    result = result[
        result[sources.class_col].isin(sources.allowed_classes)
    ].copy()
    if result.empty:
        raise ValueError("Benchmark dataset has no supported MKGF rows.")

    result = result.dropna(subset=required_columns).reset_index(drop=True)
    if result.empty:
        raise ValueError(
            "Benchmark dataset has no complete rows after NULL filtering."
        )

    result[sources.population_col] = [
        normalize_host_flag(value) for value in result[sources.population_col]
    ]
    for feature_name in sources.feature_columns:
        result[feature_name] = result[feature_name].astype(float)

    duplicate_mask = result[sources.source_id_col].duplicated(keep=False)
    if bool(duplicate_mask.any()):
        duplicate_ids = (
            result.loc[duplicate_mask, sources.source_id_col]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        sample = ", ".join(duplicate_ids[:5])
        raise ValueError(
            "Benchmark dataset contains duplicate source_id values and can "
            f"leak between train/test splits. Sample ids: {sample}"
        )

    source_sort_key = result[sources.source_id_col].astype(str)
    population_sort_key = result[sources.population_col].astype(int)
    result = (
        result.assign(
            _source_sort_key=source_sort_key,
            _population_sort_key=population_sort_key,
        )
        .sort_values(
            [sources.class_col, "_population_sort_key", "_source_sort_key"],
            ignore_index=True,
        )
        .drop(columns=["_source_sort_key", "_population_sort_key"])
    )
    return result


def validate_split_inputs(
    df_benchmark: pd.DataFrame,
    *,
    split_config: SplitConfig = DEFAULT_SPLIT_CONFIG,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.Series:
    """Проверить, что dataset подходит для стратифицированного split."""
    stratify_labels = build_stratify_labels(df_benchmark, sources=sources)
    label_counts = stratify_labels.value_counts()
    too_small_labels = label_counts[label_counts < 2]
    if not too_small_labels.empty:
        labels = ", ".join(map(str, too_small_labels.index.tolist()))
        raise ValueError(
            "Each stratify label must contain at least two rows. "
            f"Broken labels: {labels}"
        )

    n_rows = int(df_benchmark.shape[0])
    n_labels = int(label_counts.shape[0])
    n_test_rows = math.ceil(n_rows * split_config.test_size)
    if n_test_rows < n_labels:
        raise ValueError(
            "Test split is too small for stratified benchmark split: "
            f"need at least {n_labels} rows in test, got {n_test_rows}."
        )
    return stratify_labels


def split_benchmark_dataset(
    df_benchmark: pd.DataFrame,
    *,
    split_config: SplitConfig = DEFAULT_SPLIT_CONFIG,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> BenchmarkSplit:
    """Разбить benchmark dataset на детерминированные train/test части."""
    prepared = prepare_benchmark_dataset(
        df_benchmark=df_benchmark,
        sources=sources,
    )
    stratify_labels = validate_split_inputs(
        prepared,
        split_config=split_config,
        sources=sources,
    )

    train_index, test_index = train_test_split(
        prepared.index.to_numpy(),
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        shuffle=True,
        stratify=stratify_labels.to_numpy(),
    )

    train_df = prepared.loc[train_index].reset_index(drop=True)
    test_df = prepared.loc[test_index].reset_index(drop=True)
    return BenchmarkSplit(
        full_df=prepared.reset_index(drop=True),
        train_df=train_df,
        test_df=test_df,
    )


def load_benchmark_dataset(
    engine: Engine,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Загрузить полный benchmark dataset из Postgres."""
    sources = protocol.sources
    missing_relations = [
        relation_name
        for relation_name in (sources.host_view, sources.field_view)
        if not relation_exists(engine, relation_name)
    ]
    if missing_relations:
        raise RuntimeError(
            "Benchmark relations do not exist: "
            + ", ".join(missing_relations)
        )

    host_query = build_population_query(
        sources.host_view,
        is_host=True,
        sources=sources,
    )
    field_query = build_population_query(
        sources.field_view,
        is_host=False,
        sources=sources,
    )
    host_df = read_sql_frame(engine, host_query)
    field_df = read_sql_frame(engine, field_query)
    combined = pd.concat(
        [host_df, field_df],
        ignore_index=True,
        sort=False,
    )
    return prepare_benchmark_dataset(combined, sources=sources)


def load_and_split_benchmark_dataset(
    engine: Engine | None = None,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> BenchmarkSplit:
    """Загрузить benchmark dataset и сразу выполнить общий split."""
    actual_engine = engine or get_engine()
    benchmark_df = load_benchmark_dataset(
        actual_engine,
        protocol=protocol,
    )
    split = split_benchmark_dataset(
        benchmark_df,
        split_config=protocol.split,
        sources=protocol.sources,
    )
    validate_cross_validation_inputs(
        split.train_df,
        cv_config=protocol.cv,
        sources=protocol.sources,
    )
    return split


__all__ = [
    "BenchmarkSplit",
    "build_population_query",
    "build_stratify_labels",
    "get_engine",
    "load_and_split_benchmark_dataset",
    "load_benchmark_dataset",
    "make_engine_from_env",
    "prepare_benchmark_dataset",
    "read_sql_frame",
    "relation_exists",
    "split_benchmark_dataset",
    "validate_split_inputs",
]
