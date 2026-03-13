"""Тесты для data/contracts слоя comparison-layer."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
from analysis.model_comparison import (
    BenchmarkSources,
    ComparisonProtocol,
    CrossValidationConfig,
    SearchConfig,
    SplitConfig,
    build_stratify_labels,
    load_and_split_benchmark_dataset,
    prepare_benchmark_dataset,
    split_benchmark_dataset,
)
from sqlalchemy.engine import Engine


def make_benchmark_df(*, rows_per_group: int = 5) -> pd.DataFrame:
    """Собрать небольшой синтетический benchmark dataset."""
    rows: list[dict[str, object]] = []
    source_id = 1000
    class_offsets = {"M": 0.0, "K": 500.0, "G": 1000.0, "F": 1500.0}
    for spec_class, offset in class_offsets.items():
        for is_host in (False, True):
            for index in range(rows_per_group):
                rows.append(
                    {
                        "source_id": source_id,
                        "spec_class": spec_class,
                        "is_host": is_host,
                        "teff_gspphot": 3200.0 + offset + index,
                        "logg_gspphot": 4.1 + index * 0.01,
                        "radius_gspphot": 0.5 + index * 0.02,
                    }
                )
                source_id += 1
    return pd.DataFrame(rows)


def test_prepare_benchmark_dataset_rejects_duplicate_source_id() -> None:
    """Benchmark dataset не должен допускать повтор одного source_id."""
    df_benchmark = make_benchmark_df()
    df_benchmark.loc[1, "source_id"] = df_benchmark.loc[0, "source_id"]

    with pytest.raises(ValueError, match="duplicate source_id"):
        prepare_benchmark_dataset(df_benchmark)


def test_default_split_config_matches_vkr_contract() -> None:
    """Канонический split comparison-layer должен использовать 30% test."""
    assert SplitConfig().test_size == 0.30


def test_cross_validation_config_rejects_too_small_fold_count() -> None:
    """CV-контракт не должен допускать меньше двух folds."""
    with pytest.raises(ValueError, match="at least 2"):
        CrossValidationConfig(n_splits=1)


def test_search_config_rejects_unknown_refit_metric() -> None:
    """Search-контракт должен валидировать refit-метрику."""
    with pytest.raises(ValueError, match="refit_metric"):
        SearchConfig(refit_metric="unsupported_metric")  # type: ignore[arg-type]


def test_search_config_rejects_non_positive_precision_k() -> None:
    """Search-контракт не должен допускать `precision_k <= 0`."""
    with pytest.raises(ValueError, match="precision_k"):
        SearchConfig(precision_k=0)


def test_split_benchmark_dataset_is_deterministic() -> None:
    """Split должен быть воспроизводимым и сохранять все stratify-группы."""
    df_benchmark = make_benchmark_df()
    split_config = SplitConfig(test_size=0.25, random_state=7)
    sources = BenchmarkSources()

    split_a = split_benchmark_dataset(
        df_benchmark,
        split_config=split_config,
        sources=sources,
    )
    split_b = split_benchmark_dataset(
        df_benchmark,
        split_config=split_config,
        sources=sources,
    )

    assert split_a.full_df["source_id"].tolist() == split_b.full_df["source_id"].tolist()
    assert split_a.train_df["source_id"].tolist() == split_b.train_df["source_id"].tolist()
    assert split_a.test_df["source_id"].tolist() == split_b.test_df["source_id"].tolist()
    assert set(split_a.train_df["source_id"]).isdisjoint(set(split_a.test_df["source_id"]))
    assert split_a.full_df.shape[0] == split_a.train_df.shape[0] + split_a.test_df.shape[0]

    train_labels = sorted(build_stratify_labels(split_a.train_df, sources=sources).unique())
    test_labels = sorted(build_stratify_labels(split_a.test_df, sources=sources).unique())
    assert train_labels == test_labels


def test_load_and_split_benchmark_dataset_rejects_train_split_unsafe_for_cv(
    monkeypatch,
) -> None:
    """Общий loader должен рано падать, если train split не выдерживает 10 folds."""
    protocol = ComparisonProtocol(
        split=SplitConfig(test_size=0.30, random_state=42),
        cv=CrossValidationConfig(n_splits=10),
    )
    df_benchmark = make_benchmark_df(rows_per_group=10)

    monkeypatch.setattr(
        "analysis.model_comparison.data.load_benchmark_dataset",
        lambda engine, protocol: df_benchmark,
    )

    with pytest.raises(
        ValueError,
        match="Cross-validation requires at least 10 rows per stratify label",
    ):
        load_and_split_benchmark_dataset(
            engine=cast(Engine, object()),
            protocol=protocol,
        )
