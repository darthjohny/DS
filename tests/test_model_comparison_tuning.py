"""Тесты для tuning-инфраструктуры comparison-layer."""

from __future__ import annotations

import pandas as pd
import pytest
from analysis.model_comparison import (
    BenchmarkSources,
    CrossValidationConfig,
    SearchConfig,
    build_sklearn_search_scoring,
    build_stratified_kfold,
    build_stratify_labels,
    normalize_search_score,
    precision_at_k_from_proba,
    validate_cross_validation_inputs,
)
from sklearn.model_selection import StratifiedKFold


def make_cv_ready_benchmark_df() -> pd.DataFrame:
    """Собрать synthetic frame, пригодный для 10-fold CV."""
    rows: list[dict[str, object]] = []
    source_id = 20000
    class_offsets = {"M": 0.0, "K": 500.0, "G": 1000.0, "F": 1500.0}
    for spec_class, offset in class_offsets.items():
        for is_host in (False, True):
            for index in range(12):
                rows.append(
                    {
                        "source_id": source_id,
                        "spec_class": spec_class,
                        "is_host": is_host,
                        "teff_gspphot": 3200.0 + offset + index,
                        "logg_gspphot": 4.0 + index * 0.01,
                        "radius_gspphot": 0.5 + index * 0.02,
                    }
                )
                source_id += 1
    return pd.DataFrame(rows)


def test_build_stratified_kfold_uses_config_values() -> None:
    """CV helper должен строить `StratifiedKFold` по каноническому конфигу."""
    df_benchmark = make_cv_ready_benchmark_df()
    labels = build_stratify_labels(df_benchmark)
    cv = build_stratified_kfold(
        CrossValidationConfig(n_splits=10, shuffle=True, random_state=17)
    )
    expected_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=17,
    )

    actual_splits = [
        (train_idx.tolist(), test_idx.tolist())
        for train_idx, test_idx in cv.split(df_benchmark, labels)
    ]
    expected_splits = [
        (train_idx.tolist(), test_idx.tolist())
        for train_idx, test_idx in expected_cv.split(df_benchmark, labels)
    ]

    assert cv.get_n_splits() == 10
    assert actual_splits == expected_splits


def test_validate_cross_validation_inputs_accepts_cv_ready_frame() -> None:
    """Валидатор CV должен принимать frame, где все labels проходят по count."""
    df_benchmark = make_cv_ready_benchmark_df()
    sources = BenchmarkSources()

    labels = validate_cross_validation_inputs(
        df_benchmark,
        cv_config=CrossValidationConfig(n_splits=10),
        sources=sources,
    )

    assert len(labels) == len(df_benchmark)
    assert sorted(labels.unique()) == [
        "F|0",
        "F|1",
        "G|0",
        "G|1",
        "K|0",
        "K|1",
        "M|0",
        "M|1",
    ]


def test_validate_cross_validation_inputs_rejects_too_small_label() -> None:
    """Валидатор CV должен падать, если хотя бы одна stratify-группа слишком мала."""
    df_benchmark = make_cv_ready_benchmark_df()
    # Оставляем только 9 host-строк для класса M, чтобы сломать 10-fold CV.
    mask = (
        (df_benchmark["spec_class"] == "M")
        & (df_benchmark["is_host"].astype(bool))
    )
    broken_df = df_benchmark.loc[~mask].copy()
    broken_df = pd.concat(
        [broken_df, df_benchmark.loc[mask].head(9)],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Broken labels: M\\|1=9"):
        validate_cross_validation_inputs(
            broken_df,
            cv_config=CrossValidationConfig(n_splits=10),
        )


def test_build_sklearn_search_scoring_contains_all_required_metrics() -> None:
    """Scoring helper должен собирать полный набор метрик для GridSearchCV."""
    scoring = build_sklearn_search_scoring(
        SearchConfig(refit_metric="precision_at_k", precision_k=5)
    )

    assert sorted(scoring.keys()) == [
        "brier",
        "pr_auc",
        "precision_at_k",
        "roc_auc",
    ]


def test_precision_at_k_from_proba_uses_positive_class_column() -> None:
    """precision@k scorer должен брать positive-class вероятности из `predict_proba`."""
    score = precision_at_k_from_proba(
        [1, 0, 1, 0],
        [
            [0.10, 0.90],
            [0.80, 0.20],
            [0.35, 0.65],
            [0.60, 0.40],
        ],
        k=2,
    )

    assert score == 1.0


def test_normalize_search_score_inverts_negative_brier() -> None:
    """User-facing best score для Brier должен возвращаться в обычной шкале."""
    assert normalize_search_score(-0.125, metric="brier") == 0.125
