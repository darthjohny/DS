"""Manual cross-validation search для non-sklearn comparison-моделей.

Модуль нужен для тех baseline/model wrapper-ов, которые не являются
sklearn-estimator из коробки, но всё равно должны соблюдать единый
benchmark-контракт ВКР:

- train/test split задаётся снаружи;
- подбор параметров идёт только на train-части;
- внутри train используется канонический stratified CV;
- итоговая модель переобучается на всём train split после выбора best params.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd
from sklearn.model_selection import ParameterGrid

from analysis.model_comparison.contracts import (
    DEFAULT_BENCHMARK_SOURCES,
    DEFAULT_CV_CONFIG,
    DEFAULT_SEARCH_CONFIG,
    BenchmarkSources,
    CrossValidationConfig,
    ModelSearchSummary,
    SearchConfig,
    SearchRefitMetric,
)
from analysis.model_comparison.metrics import (
    precision_at_k,
    safe_brier_score,
    safe_pr_auc,
    safe_roc_auc,
    validate_scored_frame,
)
from analysis.model_comparison.tuning import build_stratified_kfold, build_stratify_labels


def evaluate_search_metric(
    scored_df: pd.DataFrame,
    *,
    metric: SearchRefitMetric,
    precision_k: int,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> float:
    """Посчитать search-метрику по scored validation frame."""
    validated = validate_scored_frame(scored_df, sources=sources)
    y_true = validated[sources.population_col].astype(int)
    y_score = validated["model_score"]

    if metric == "roc_auc":
        return safe_roc_auc(y_true, y_score)
    if metric == "pr_auc":
        return safe_pr_auc(y_true, y_score)
    if metric == "brier":
        return safe_brier_score(y_true, y_score)
    precision_value, _ = precision_at_k(
        validated,
        k=precision_k,
        sources=sources,
    )
    return precision_value


def rank_search_metric(score_value: float, *, metric: SearchRefitMetric) -> float:
    """Преобразовать метрику к шкале, где большее значение лучше."""
    if metric == "brier":
        return -float(score_value)
    return float(score_value)


def run_manual_model_search[ModelT](
    df_train: pd.DataFrame,
    *,
    model_name: str,
    param_grid: dict[str, list[object]],
    fit_model: Callable[[pd.DataFrame, Mapping[str, object]], ModelT],
    score_model: Callable[[ModelT, pd.DataFrame], pd.DataFrame],
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[ModelT, ModelSearchSummary]:
    """Запустить manual CV search и вернуть лучшую модель с summary."""
    candidates = list(ParameterGrid(param_grid))
    if not candidates:
        raise ValueError("Manual search requires a non-empty parameter grid.")

    validated = df_train.reset_index(drop=True).copy()
    stratify_labels = build_stratify_labels(validated, sources=sources)
    splitter = build_stratified_kfold(cv_config=cv_config)

    best_params: dict[str, object] | None = None
    best_raw_score: float | None = None
    best_rank_score: float | None = None
    last_error: ValueError | None = None

    for candidate_params in candidates:
        fold_scores: list[float] = []
        candidate_failed = False

        for train_index, valid_index in splitter.split(validated, stratify_labels):
            fold_train_df = validated.iloc[train_index].reset_index(drop=True)
            fold_valid_df = validated.iloc[valid_index].reset_index(drop=True)
            try:
                fold_model = fit_model(fold_train_df, candidate_params)
                fold_scored_df = score_model(fold_model, fold_valid_df)
            except ValueError as exc:
                last_error = exc
                candidate_failed = True
                break

            fold_scores.append(
                evaluate_search_metric(
                    fold_scored_df,
                    metric=search_config.refit_metric,
                    precision_k=search_config.precision_k,
                    sources=sources,
                )
            )

        if candidate_failed or not fold_scores:
            continue

        mean_score = float(sum(fold_scores) / len(fold_scores))
        rank_score = rank_search_metric(
            mean_score,
            metric=search_config.refit_metric,
        )
        if best_rank_score is None or rank_score > best_rank_score:
            best_params = {str(key): value for key, value in candidate_params.items()}
            best_raw_score = mean_score
            best_rank_score = rank_score

    if best_params is None or best_raw_score is None:
        message = "Manual search found no valid candidate."
        if last_error is not None:
            message += f" Last error: {last_error}"
        raise ValueError(message)

    best_model = fit_model(validated, best_params)
    search_summary = ModelSearchSummary(
        model_name=model_name,
        refit_metric=search_config.refit_metric,
        precision_k=search_config.precision_k,
        cv_folds=cv_config.n_splits,
        n_train_rows=int(validated.shape[0]),
        n_host=int(validated[sources.population_col].astype(bool).sum()),
        n_field=int((~validated[sources.population_col].astype(bool)).sum()),
        candidate_count=len(candidates),
        best_cv_score=float(best_raw_score),
        best_params=best_params,
    )
    return best_model, search_summary


__all__ = [
    "evaluate_search_metric",
    "rank_search_metric",
    "run_manual_model_search",
]
