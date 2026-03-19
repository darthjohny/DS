"""Cross-validation helpers для comparison-layer.

Что делает модуль:
    - собирает единые stratify-метки для benchmark dataset;
    - строит канонический `StratifiedKFold` для tuning-контура;
    - проверяет, что train split подходит для заданного числа folds.
    - строит общий набор sklearn-scorer-ов для model search.

Где используется:
    - в data-layer для ранней валидации benchmark split;
    - в model wrapper-ах для единообразного CV-контракта.

Что модуль не делает:
    - не обучает модели;
    - не считает supervised-метрики;
    - не пишет артефакты на диск.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold

from analysis.model_comparison.contracts import (
    DEFAULT_BENCHMARK_SOURCES,
    DEFAULT_CV_CONFIG,
    DEFAULT_SEARCH_CONFIG,
    BenchmarkSources,
    CrossValidationConfig,
    SearchConfig,
    SearchRefitMetric,
)


def build_stratify_labels(
    df_benchmark: pd.DataFrame,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.Series:
    """Собрать stratify-метки вида `spec_class|host_or_field`."""
    return (
        df_benchmark[sources.class_col].astype(str)
        + "|"
        + df_benchmark[sources.population_col].astype(int).astype(str)
    )


def build_stratified_kfold(
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
) -> StratifiedKFold:
    """Построить канонический `StratifiedKFold` для tuning-контура."""
    random_state = cv_config.random_state if cv_config.shuffle else None
    return StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=random_state,
    )


def positive_class_scores(y_score: Any) -> np.ndarray:
    """Нормализовать scorer output до одномерного массива positive-class score."""
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim == 1:
        return scores
    if scores.ndim == 2 and scores.shape[1] == 1:
        return scores[:, 0]
    if scores.ndim == 2 and scores.shape[1] == 2:
        return scores[:, 1]
    raise ValueError(
        "Expected binary classifier scores with shape (n,) or (n, 2), "
        f"got {scores.shape}."
    )


def brier_score_from_proba(y_true: Any, y_score: Any) -> float:
    """Посчитать Brier score по positive-class вероятностям."""
    return float(
        brier_score_loss(
            np.asarray(y_true, dtype=int),
            positive_class_scores(y_score),
        )
    )


def precision_at_k_from_proba(
    y_true: Any,
    y_score: Any,
    *,
    k: int,
) -> float:
    """Посчитать precision@k по positive-class вероятностям."""
    scores = positive_class_scores(y_score)
    labels = np.asarray(y_true, dtype=float)
    effective_k = min(int(k), int(scores.shape[0]))
    if effective_k <= 0:
        raise ValueError("precision_at_k_from_proba expects at least one row.")

    ranked_index = np.argsort(-scores, kind="mergesort")
    top_labels = labels[ranked_index[:effective_k]]
    return float(np.mean(top_labels))


def build_sklearn_search_scoring(
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> dict[str, str | Any]:
    """Собрать единый scoring dict для `GridSearchCV` в comparison-layer."""
    return {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "brier": make_scorer(
            brier_score_from_proba,
            response_method="predict_proba",
            greater_is_better=False,
        ),
        "precision_at_k": make_scorer(
            precision_at_k_from_proba,
            response_method="predict_proba",
            greater_is_better=True,
            k=search_config.precision_k,
        ),
    }


def normalize_search_score(
    score_value: float,
    *,
    metric: SearchRefitMetric,
) -> float:
    """Привести best score из search-объекта к пользовательской шкале метрики."""
    if metric == "brier":
        return -float(score_value)
    return float(score_value)


def extract_best_cv_score_stats(search: Any, *, metric: SearchRefitMetric) -> tuple[float, float, float, float]:
    """Достать mean/std/min/max fold-score для лучшей search-конфигурации."""
    if not hasattr(search, "best_index_") or not hasattr(search, "cv_results_"):
        raise TypeError("Search object must expose best_index_ and cv_results_.")

    best_index = int(search.best_index_)
    cv_results = search.cv_results_
    score_key = f"mean_test_{metric}"
    std_key = f"std_test_{metric}"
    split_prefix = "_test_"

    if score_key not in cv_results or std_key not in cv_results:
        raise KeyError(f"Search results are missing keys: {score_key}, {std_key}.")

    mean_score = normalize_search_score(float(cv_results[score_key][best_index]), metric=metric)
    std_score = float(cv_results[std_key][best_index])

    fold_scores: list[float] = []
    split_keys = sorted(
        key
        for key in cv_results
        if key.startswith("split") and split_prefix + metric in key
    )
    for key in split_keys:
        fold_scores.append(
            normalize_search_score(float(cv_results[key][best_index]), metric=metric)
        )

    if not fold_scores:
        raise ValueError("Search results did not expose per-fold test scores.")

    return (
        mean_score,
        std_score,
        float(min(fold_scores)),
        float(max(fold_scores)),
    )


def validate_cross_validation_inputs(
    df_benchmark: pd.DataFrame,
    *,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.Series:
    """Проверить, что frame подходит для stratified cross-validation.

    Источник данных
    ---------------
    Ожидает уже подготовленный benchmark frame или train split, в котором
    присутствуют колонки `spec_class` и `is_host` из comparison-контракта.
    """
    if df_benchmark.empty:
        raise ValueError(
            "Cross-validation requires a non-empty benchmark frame."
        )

    stratify_labels = build_stratify_labels(df_benchmark, sources=sources)
    label_counts = stratify_labels.value_counts()
    too_small_labels = label_counts[label_counts < cv_config.n_splits]
    if too_small_labels.empty:
        return stratify_labels

    labels_with_counts = ", ".join(
        f"{label}={int(count)}"
        for label, count in too_small_labels.items()
    )
    raise ValueError(
        "Cross-validation requires at least "
        f"{cv_config.n_splits} rows per stratify label. "
        f"Broken labels: {labels_with_counts}."
    )


__all__ = [
    "brier_score_from_proba",
    "extract_best_cv_score_stats",
    "build_stratified_kfold",
    "build_stratify_labels",
    "build_sklearn_search_scoring",
    "normalize_search_score",
    "positive_class_scores",
    "precision_at_k_from_proba",
    "validate_cross_validation_inputs",
]
