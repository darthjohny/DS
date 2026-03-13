"""Wrapper основной contrastive-модели для comparison-layer.

Модуль переиспользует текущий production training/scoring path host-модели,
но запускает его на общем benchmark split comparison-layer. Это позволяет
сравнивать `main_contrastive_v1` с baseline-моделями на одном и том же
train/test наборе.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from analysis.model_comparison.contracts import (
    DEFAULT_BENCHMARK_SOURCES,
    DEFAULT_CV_CONFIG,
    DEFAULT_SEARCH_CONFIG,
    BenchmarkSources,
    CrossValidationConfig,
    ModelScoreFrames,
    ModelSearchSummary,
    SearchConfig,
)
from analysis.model_comparison.data import BenchmarkSplit, prepare_benchmark_dataset
from analysis.model_comparison.manual_search import run_manual_model_search
from host_model.artifacts import ContrastiveGaussianModel
from host_model.contrastive_score import score_df_contrastive
from host_model.fit import fit_contrastive_gaussian_model

MAIN_CONTRASTIVE_MODEL_NAME = "main_contrastive_v1"


@dataclass(slots=True)
class ContrastiveModelRun:
    """Результат одного comparison-прогона основной contrastive-модели."""

    model: ContrastiveGaussianModel
    search_summary: ModelSearchSummary
    scored_split: ModelScoreFrames


def validate_benchmark_frame(
    df_benchmark: pd.DataFrame,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Проверить benchmark frame перед contrastive training/scoring."""
    return prepare_benchmark_dataset(df_benchmark, sources=sources)


def fit_main_contrastive_model(
    df_train: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    use_m_subclasses: bool = True,
    shrink_alpha: float = 0.15,
    min_population_size: int = 2,
) -> ContrastiveGaussianModel:
    """Обучить основную contrastive-модель на benchmark train split."""
    validated = validate_benchmark_frame(df_train, sources=sources)
    return fit_contrastive_gaussian_model(
        validated,
        population_col=sources.population_col,
        use_m_subclasses=use_m_subclasses,
        shrink_alpha=shrink_alpha,
        min_population_size=min_population_size,
    )


def build_contrastive_param_grid() -> dict[str, list[object]]:
    """Собрать компактный search-space для основной contrastive-модели."""
    return {
        "use_m_subclasses": [False, True],
        "shrink_alpha": [0.05, 0.15],
        "min_population_size": [2, 4],
    }


def fit_main_contrastive_model_with_search(
    df_train: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[ContrastiveGaussianModel, ModelSearchSummary]:
    """Подобрать и обучить лучшую contrastive-модель по train split."""
    validated = validate_benchmark_frame(df_train, sources=sources)
    return run_manual_model_search(
        validated,
        model_name=MAIN_CONTRASTIVE_MODEL_NAME,
        param_grid=build_contrastive_param_grid(),
        fit_model=lambda candidate_df, params: fit_main_contrastive_model(
            candidate_df,
            sources=sources,
            use_m_subclasses=bool(params["use_m_subclasses"]),
            shrink_alpha=float(cast(float, params["shrink_alpha"])),
            min_population_size=int(cast(int, params["min_population_size"])),
        ),
        score_model=lambda candidate_model, candidate_df: score_main_contrastive_model(
            candidate_model,
            candidate_df,
            sources=sources,
        ),
        sources=sources,
        cv_config=cv_config,
        search_config=search_config,
    )


def score_main_contrastive_model(
    model: ContrastiveGaussianModel,
    df_part: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Скорить benchmark frame текущей contrastive-моделью."""
    validated = validate_benchmark_frame(df_part, sources=sources)
    scored = score_df_contrastive(
        model=model,
        df=validated,
        spec_class_col=sources.class_col,
    ).copy()
    scored["model_name"] = MAIN_CONTRASTIVE_MODEL_NAME
    scored["model_score"] = scored["host_posterior"].astype(float)
    return scored


def run_main_contrastive_model(
    split: BenchmarkSplit,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> ContrastiveModelRun:
    """Обучить и проскорить contrastive-модель на общем benchmark split."""
    model, search_summary = fit_main_contrastive_model_with_search(
        split.train_df,
        sources=sources,
        cv_config=cv_config,
        search_config=search_config,
    )
    scored_split = ModelScoreFrames(
        model_name=MAIN_CONTRASTIVE_MODEL_NAME,
        train_scored_df=score_main_contrastive_model(
            model=model,
            df_part=split.train_df,
            sources=sources,
        ),
        test_scored_df=score_main_contrastive_model(
            model=model,
            df_part=split.test_df,
            sources=sources,
        ),
    )
    return ContrastiveModelRun(
        model=model,
        search_summary=search_summary,
        scored_split=scored_split,
    )


__all__ = [
    "ContrastiveModelRun",
    "MAIN_CONTRASTIVE_MODEL_NAME",
    "build_contrastive_param_grid",
    "fit_main_contrastive_model",
    "fit_main_contrastive_model_with_search",
    "run_main_contrastive_model",
    "score_main_contrastive_model",
    "validate_benchmark_frame",
]
