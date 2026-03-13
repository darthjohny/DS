"""Wrapper второго baseline: class-specific RandomForest.

Модуль реализует воспроизводимый ML baseline для задачи `host vs field`:

- одна `RandomForestClassifier` на каждый класс `M/K/G/F`;
- выбор гиперпараметров идёт внутри train split через `GridSearchCV`;
- score наружу выдаётся через единый `model_score`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid

from analysis.model_comparison.contracts import (
    DEFAULT_BENCHMARK_SOURCES,
    DEFAULT_CV_CONFIG,
    DEFAULT_RANDOM_FOREST_CONFIG,
    DEFAULT_SEARCH_CONFIG,
    BenchmarkSources,
    ClassSearchSummary,
    CrossValidationConfig,
    ModelScoreFrames,
    RandomForestConfig,
    SearchConfig,
)
from analysis.model_comparison.data import BenchmarkSplit, prepare_benchmark_dataset
from analysis.model_comparison.tuning import (
    build_sklearn_search_scoring,
    build_stratified_kfold,
    normalize_search_score,
)

RANDOM_FOREST_MODEL_NAME = "baseline_random_forest"


@dataclass(slots=True)
class RandomForestBaselineRun:
    """Результат одного baseline-прогона для class-specific RandomForest."""

    models_by_class: dict[str, RandomForestClassifier]
    search_results_by_class: dict[str, ClassSearchSummary]
    scored_split: ModelScoreFrames


def validate_benchmark_frame(
    df_benchmark: pd.DataFrame,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Проверить benchmark frame перед RandomForest training/scoring."""
    return prepare_benchmark_dataset(df_benchmark, sources=sources)


def validate_class_training_rows(
    class_df: pd.DataFrame,
    *,
    spec_class: str,
    sources: BenchmarkSources,
) -> None:
    """Проверить, что class-specific train frame пригоден для обучения RF."""
    if class_df.empty:
        raise ValueError(
            f"RandomForest baseline received no rows for class {spec_class}."
        )

    label_counts = class_df[sources.population_col].value_counts()
    if int(label_counts.shape[0]) < 2:
        raise ValueError(
            "RandomForest baseline requires both host and field rows "
            f"for class {spec_class}."
        )


def build_random_forest_estimator(
    config: RandomForestConfig = DEFAULT_RANDOM_FOREST_CONFIG,
) -> RandomForestClassifier:
    """Собрать base estimator для RandomForest baseline."""
    return RandomForestClassifier(
        random_state=config.random_state,
        class_weight=config.class_weight,
        n_jobs=config.n_jobs,
    )


def build_random_forest_param_grid(
    config: RandomForestConfig = DEFAULT_RANDOM_FOREST_CONFIG,
) -> dict[str, list[int]]:
    """Собрать компактный search-space для `GridSearchCV` RandomForest."""
    n_estimators_values = sorted(
        {
            max(100, int(config.n_estimators) // 3),
            int(config.n_estimators),
        }
    )
    min_samples_leaf_values = sorted(
        {
            1,
            int(config.min_samples_leaf),
            max(4, int(config.min_samples_leaf) * 2),
        }
    )
    return {
        "n_estimators": n_estimators_values,
        "min_samples_leaf": min_samples_leaf_values,
    }


def search_random_forest_for_class(
    class_df: pd.DataFrame,
    *,
    spec_class: str,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    config: RandomForestConfig = DEFAULT_RANDOM_FOREST_CONFIG,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[RandomForestClassifier, ClassSearchSummary]:
    """Подобрать лучший RandomForest для одного спектрального класса."""
    validate_class_training_rows(
        class_df,
        spec_class=spec_class,
        sources=sources,
    )

    estimator = build_random_forest_estimator(config=config)
    param_grid = build_random_forest_param_grid(config=config)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=build_sklearn_search_scoring(search_config=search_config),
        refit=search_config.refit_metric,
        cv=build_stratified_kfold(cv_config=cv_config),
        n_jobs=1,
    )
    search.fit(
        class_df[list(sources.feature_columns)].to_numpy(),
        class_df[sources.population_col].astype(bool).to_numpy(),
    )

    best_model = search.best_estimator_
    if not isinstance(best_model, RandomForestClassifier):
        raise TypeError(
            "RandomForest baseline expected GridSearchCV.best_estimator_ "
            "to be RandomForestClassifier."
        )

    search_summary = ClassSearchSummary(
        model_name=RANDOM_FOREST_MODEL_NAME,
        spec_class=str(spec_class),
        refit_metric=search_config.refit_metric,
        precision_k=search_config.precision_k,
        cv_folds=cv_config.n_splits,
        n_train_rows=int(class_df.shape[0]),
        n_host=int(class_df[sources.population_col].astype(bool).sum()),
        n_field=int((~class_df[sources.population_col].astype(bool)).sum()),
        candidate_count=len(ParameterGrid(param_grid)),
        best_cv_score=normalize_search_score(
            float(search.best_score_),
            metric=search_config.refit_metric,
        ),
        best_params={str(key): value for key, value in search.best_params_.items()},
    )
    return best_model, search_summary


def fit_random_forest_baseline(
    df_train: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    config: RandomForestConfig = DEFAULT_RANDOM_FOREST_CONFIG,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> dict[str, RandomForestClassifier]:
    """Обучить class-specific RandomForest модели по train split."""
    models_by_class, _ = fit_random_forest_baseline_with_search(
        df_train,
        sources=sources,
        config=config,
        cv_config=cv_config,
        search_config=search_config,
    )
    return models_by_class


def fit_random_forest_baseline_with_search(
    df_train: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    config: RandomForestConfig = DEFAULT_RANDOM_FOREST_CONFIG,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[dict[str, RandomForestClassifier], dict[str, ClassSearchSummary]]:
    """Обучить RF baseline и вернуть модели вместе с search summary."""
    validated = validate_benchmark_frame(df_train, sources=sources)
    models_by_class: dict[str, RandomForestClassifier] = {}
    search_results_by_class: dict[str, ClassSearchSummary] = {}

    for spec_class in sources.allowed_classes:
        class_df = validated[
            validated[sources.class_col] == spec_class
        ].reset_index(drop=True)
        best_model, search_summary = search_random_forest_for_class(
            class_df,
            spec_class=spec_class,
            sources=sources,
            config=config,
            cv_config=cv_config,
            search_config=search_config,
        )
        models_by_class[spec_class] = best_model
        search_results_by_class[spec_class] = search_summary

    return models_by_class, search_results_by_class


def score_random_forest_baseline(
    models_by_class: dict[str, RandomForestClassifier],
    df_part: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Скорить benchmark frame через class-specific RandomForest."""
    validated = validate_benchmark_frame(df_part, sources=sources)
    scored_parts: list[pd.DataFrame] = []

    for spec_class in sources.allowed_classes:
        class_df = validated[validated[sources.class_col] == spec_class].copy()
        if class_df.empty:
            continue
        if spec_class not in models_by_class:
            raise ValueError(
                "RandomForest baseline has no fitted model for class "
                f"{spec_class}."
            )

        estimator = models_by_class[spec_class]
        features = class_df[list(sources.feature_columns)].to_numpy()
        positive_proba = estimator.predict_proba(features)[:, 1]
        predicted_host = estimator.predict(features)

        class_df["model_name"] = RANDOM_FOREST_MODEL_NAME
        class_df["model_score"] = positive_proba.astype(float)
        class_df["rf_positive_proba"] = positive_proba.astype(float)
        class_df["rf_predicted_is_host"] = predicted_host.astype(bool)
        scored_parts.append(class_df)

    if not scored_parts:
        raise ValueError("RandomForest baseline produced no scored rows.")

    return pd.concat(
        scored_parts,
        ignore_index=True,
        sort=False,
    )


def run_random_forest_baseline(
    split: BenchmarkSplit,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    config: RandomForestConfig = DEFAULT_RANDOM_FOREST_CONFIG,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> RandomForestBaselineRun:
    """Обучить и проскорить RandomForest baseline на общем benchmark split."""
    models_by_class, search_results_by_class = fit_random_forest_baseline_with_search(
        split.train_df,
        sources=sources,
        config=config,
        cv_config=cv_config,
        search_config=search_config,
    )
    scored_split = ModelScoreFrames(
        model_name=RANDOM_FOREST_MODEL_NAME,
        train_scored_df=score_random_forest_baseline(
            models_by_class,
            split.train_df,
            sources=sources,
        ),
        test_scored_df=score_random_forest_baseline(
            models_by_class,
            split.test_df,
            sources=sources,
        ),
    )
    return RandomForestBaselineRun(
        models_by_class=models_by_class,
        search_results_by_class=search_results_by_class,
        scored_split=scored_split,
    )


__all__ = [
    "RANDOM_FOREST_MODEL_NAME",
    "RandomForestBaselineRun",
    "build_random_forest_estimator",
    "build_random_forest_param_grid",
    "fit_random_forest_baseline",
    "fit_random_forest_baseline_with_search",
    "run_random_forest_baseline",
    "score_random_forest_baseline",
    "search_random_forest_for_class",
    "validate_benchmark_frame",
    "validate_class_training_rows",
]
