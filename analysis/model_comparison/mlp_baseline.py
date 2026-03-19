"""Wrapper компактного class-specific MLP baseline.

Модуль реализует минимальный ИНС-baseline для задачи `host vs field`:

- одна маленькая `MLPClassifier` на каждый класс `M/K/G/F`;
- перед сетью обязательно применяется `StandardScaler`;
- наружу возвращается тот же общий `model_score`, что и у остальных
  comparison-wrapper-ов.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import cast

import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.model_comparison.contracts import (
    DEFAULT_BENCHMARK_SOURCES,
    DEFAULT_CV_CONFIG,
    DEFAULT_MLP_BASELINE_CONFIG,
    DEFAULT_SEARCH_CONFIG,
    BenchmarkSources,
    ClassSearchSummary,
    CrossValidationConfig,
    MLPBaselineConfig,
    ModelScoreFrames,
    SearchConfig,
)
from analysis.model_comparison.data import BenchmarkSplit, prepare_benchmark_dataset
from analysis.model_comparison.tuning import (
    build_sklearn_search_scoring,
    build_stratified_kfold,
    extract_best_cv_score_stats,
)

MLP_BASELINE_MODEL_NAME = "baseline_mlp_small"


@dataclass(slots=True)
class MLPBaselineRun:
    """Результат одного baseline-прогона для class-specific MLP."""

    models_by_class: dict[str, Pipeline]
    search_results_by_class: dict[str, ClassSearchSummary]
    scored_split: ModelScoreFrames


def validate_benchmark_frame(
    df_benchmark: pd.DataFrame,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Проверить benchmark frame перед MLP training/scoring."""
    return prepare_benchmark_dataset(df_benchmark, sources=sources)


def use_early_stopping(
    class_df: pd.DataFrame,
    *,
    sources: BenchmarkSources,
    config: MLPBaselineConfig,
    sample_size: int | None = None,
) -> bool:
    """Определить, достаточно ли данных для безопасного early stopping.

    На маленьких class-specific train выборках `MLPClassifier` может
    сломаться на внутреннем validation split. В таких случаях baseline
    должен автоматически перейти в более простой и стабильный режим.
    """
    if not config.early_stopping:
        return False

    n_rows = int(sample_size) if sample_size is not None else int(class_df.shape[0])
    n_classes = int(class_df[sources.population_col].nunique(dropna=False))
    min_validation_rows = max(2, n_classes)
    validation_rows = int(round(n_rows * config.validation_fraction))
    return validation_rows >= min_validation_rows


def estimate_cv_train_sample_size(
    sample_size: int,
    *,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
) -> int:
    """Оценить минимальный размер train-fold для class-specific search."""
    held_out_rows = ceil(int(sample_size) / int(cv_config.n_splits))
    return max(1, int(sample_size) - held_out_rows)


def build_mlp_pipeline(
    *,
    config: MLPBaselineConfig = DEFAULT_MLP_BASELINE_CONFIG,
    early_stopping: bool,
    sample_size: int,
) -> Pipeline:
    """Собрать стандартный `StandardScaler + MLPClassifier` pipeline."""
    effective_batch_size = min(int(config.batch_size), int(sample_size))
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=config.hidden_layer_sizes,
                    activation=config.activation,
                    solver=config.solver,
                    alpha=config.alpha,
                    learning_rate_init=config.learning_rate_init,
                    batch_size=effective_batch_size,
                    max_iter=config.max_iter,
                    tol=config.tol,
                    early_stopping=early_stopping,
                    validation_fraction=config.validation_fraction,
                    n_iter_no_change=config.n_iter_no_change,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def get_mlp_classifier(estimator: Pipeline, *, source: str) -> MLPClassifier:
    """Достать типизированный `MLPClassifier` из sklearn pipeline."""
    mlp = estimator.named_steps.get("mlp")
    if not isinstance(mlp, MLPClassifier):
        raise TypeError(f"{source} expected sklearn MLPClassifier in pipeline step 'mlp'.")
    return cast(MLPClassifier, mlp)


def validate_class_training_rows(
    class_df: pd.DataFrame,
    *,
    spec_class: str,
    sources: BenchmarkSources,
) -> None:
    """Проверить, что class-specific train frame пригоден для обучения MLP."""
    if class_df.empty:
        raise ValueError(f"MLP baseline received no rows for class {spec_class}.")

    label_counts = class_df[sources.population_col].value_counts()
    if int(label_counts.shape[0]) < 2:
        raise ValueError(
            "MLP baseline requires both host and field rows "
            f"for class {spec_class}."
        )


def build_mlp_param_grid(
    config: MLPBaselineConfig = DEFAULT_MLP_BASELINE_CONFIG,
) -> dict[str, list[object]]:
    """Собрать компактный search-space для `GridSearchCV` MLP baseline."""
    compact_hidden_sizes = tuple(
        max(4, int(size) // 2) for size in config.hidden_layer_sizes
    )
    hidden_layer_sizes_values = list(
        dict.fromkeys(
            [
                compact_hidden_sizes,
                tuple(int(size) for size in config.hidden_layer_sizes),
            ]
        )
    )
    alpha_values = sorted(
        {
            max(0.0001, float(config.alpha) / 10.0),
            float(config.alpha),
            float(config.alpha) * 10.0,
        }
    )
    hidden_layer_param_values: list[object] = [
        value for value in hidden_layer_sizes_values
    ]
    alpha_param_values: list[object] = [value for value in alpha_values]
    return {
        "mlp__hidden_layer_sizes": hidden_layer_param_values,
        "mlp__alpha": alpha_param_values,
    }


def search_mlp_for_class(
    class_df: pd.DataFrame,
    *,
    spec_class: str,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    config: MLPBaselineConfig = DEFAULT_MLP_BASELINE_CONFIG,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[Pipeline, ClassSearchSummary]:
    """Подобрать лучший MLP pipeline для одного спектрального класса."""
    validate_class_training_rows(
        class_df,
        spec_class=spec_class,
        sources=sources,
    )

    cv_train_sample_size = estimate_cv_train_sample_size(
        int(class_df.shape[0]),
        cv_config=cv_config,
    )
    estimator = build_mlp_pipeline(
        config=config,
        early_stopping=use_early_stopping(
            class_df,
            sources=sources,
            config=config,
            sample_size=cv_train_sample_size,
        ),
        sample_size=cv_train_sample_size,
    )
    param_grid = build_mlp_param_grid(config=config)
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

    best_pipeline = search.best_estimator_
    if not isinstance(best_pipeline, Pipeline):
        raise TypeError(
            "MLP baseline expected GridSearchCV.best_estimator_ "
            "to be sklearn Pipeline."
        )

    best_cv_score, cv_score_std, cv_score_min, cv_score_max = extract_best_cv_score_stats(
        search,
        metric=search_config.refit_metric,
    )
    search_summary = ClassSearchSummary(
        model_name=MLP_BASELINE_MODEL_NAME,
        spec_class=str(spec_class),
        refit_metric=search_config.refit_metric,
        precision_k=search_config.precision_k,
        cv_folds=cv_config.n_splits,
        n_train_rows=int(class_df.shape[0]),
        n_host=int(class_df[sources.population_col].astype(bool).sum()),
        n_field=int((~class_df[sources.population_col].astype(bool)).sum()),
        candidate_count=len(ParameterGrid(param_grid)),
        best_cv_score=best_cv_score,
        cv_score_std=cv_score_std,
        cv_score_min=cv_score_min,
        cv_score_max=cv_score_max,
        best_params={
            str(key).removeprefix("mlp__"): value
            for key, value in search.best_params_.items()
        },
    )
    return best_pipeline, search_summary


def fit_mlp_baseline(
    df_train: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    config: MLPBaselineConfig = DEFAULT_MLP_BASELINE_CONFIG,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> dict[str, Pipeline]:
    """Обучить class-specific MLP модели по train split."""
    models_by_class, _ = fit_mlp_baseline_with_search(
        df_train,
        sources=sources,
        config=config,
        cv_config=cv_config,
        search_config=search_config,
    )
    return models_by_class


def fit_mlp_baseline_with_search(
    df_train: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    config: MLPBaselineConfig = DEFAULT_MLP_BASELINE_CONFIG,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[dict[str, Pipeline], dict[str, ClassSearchSummary]]:
    """Обучить MLP baseline и вернуть модели вместе с search summary."""
    validated = validate_benchmark_frame(df_train, sources=sources)
    models_by_class: dict[str, Pipeline] = {}
    search_results_by_class: dict[str, ClassSearchSummary] = {}

    for spec_class in sources.allowed_classes:
        class_df = validated[
            validated[sources.class_col] == spec_class
        ].reset_index(drop=True)
        best_pipeline, search_summary = search_mlp_for_class(
            class_df,
            spec_class=spec_class,
            sources=sources,
            config=config,
            cv_config=cv_config,
            search_config=search_config,
        )
        models_by_class[spec_class] = best_pipeline
        search_results_by_class[spec_class] = search_summary

    return models_by_class, search_results_by_class


def score_mlp_baseline(
    models_by_class: dict[str, Pipeline],
    df_part: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Скорить benchmark frame через class-specific MLP."""
    validated = validate_benchmark_frame(df_part, sources=sources)
    scored_parts: list[pd.DataFrame] = []

    for spec_class in sources.allowed_classes:
        class_df = validated[validated[sources.class_col] == spec_class].copy()
        if class_df.empty:
            continue
        if spec_class not in models_by_class:
            raise ValueError(
                "MLP baseline has no fitted model for class "
                f"{spec_class}."
            )

        estimator = models_by_class[spec_class]
        features = class_df[list(sources.feature_columns)].to_numpy()
        positive_proba = estimator.predict_proba(features)[:, 1]
        predicted_host = estimator.predict(features)

        class_df["model_name"] = MLP_BASELINE_MODEL_NAME
        class_df["model_score"] = positive_proba.astype(float)
        class_df["mlp_positive_proba"] = positive_proba.astype(float)
        class_df["mlp_predicted_is_host"] = [bool(value) for value in predicted_host]
        scored_parts.append(class_df)

    if not scored_parts:
        raise ValueError("MLP baseline produced no scored rows.")

    return pd.concat(
        scored_parts,
        ignore_index=True,
        sort=False,
    )


def run_mlp_baseline(
    split: BenchmarkSplit,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    config: MLPBaselineConfig = DEFAULT_MLP_BASELINE_CONFIG,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> MLPBaselineRun:
    """Обучить и проскорить MLP baseline на общем benchmark split."""
    models_by_class, search_results_by_class = fit_mlp_baseline_with_search(
        split.train_df,
        sources=sources,
        config=config,
        cv_config=cv_config,
        search_config=search_config,
    )
    scored_split = ModelScoreFrames(
        model_name=MLP_BASELINE_MODEL_NAME,
        train_scored_df=score_mlp_baseline(
            models_by_class,
            split.train_df,
            sources=sources,
        ),
        test_scored_df=score_mlp_baseline(
            models_by_class,
            split.test_df,
            sources=sources,
        ),
    )
    return MLPBaselineRun(
        models_by_class=models_by_class,
        search_results_by_class=search_results_by_class,
        scored_split=scored_split,
    )


__all__ = [
    "MLP_BASELINE_MODEL_NAME",
    "MLPBaselineRun",
    "build_mlp_param_grid",
    "build_mlp_pipeline",
    "estimate_cv_train_sample_size",
    "fit_mlp_baseline",
    "fit_mlp_baseline_with_search",
    "get_mlp_classifier",
    "run_mlp_baseline",
    "search_mlp_for_class",
    "score_mlp_baseline",
    "use_early_stopping",
    "validate_class_training_rows",
    "validate_benchmark_frame",
]
