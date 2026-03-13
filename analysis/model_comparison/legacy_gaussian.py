"""Wrapper первого baseline: legacy Gaussian similarity.

Модуль не содержит собственной математики. Он аккуратно переиспользует
legacy-контур из `src/host_model` и приводит его к comparison-contract:

- обучение идёт только на host-строках train split;
- score считается на train/test частях общего benchmark;
- наружу возвращается единый `model_score`, совместимый с будущими метриками.
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
from host_model.artifacts import LegacyGaussianModel
from host_model.fit import fit_gaussian_model
from host_model.legacy_score import score_df

LEGACY_GAUSSIAN_MODEL_NAME = "baseline_legacy_gaussian"


@dataclass(slots=True)
class LegacyGaussianBaselineRun:
    """Результат одного baseline-прогона для legacy Gaussian."""

    model: LegacyGaussianModel
    search_summary: ModelSearchSummary
    scored_split: ModelScoreFrames


def validate_benchmark_frame(
    df_benchmark: pd.DataFrame,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Проверить benchmark frame перед legacy training/scoring."""
    return prepare_benchmark_dataset(df_benchmark, sources=sources)


def select_host_training_rows(
    df_benchmark: pd.DataFrame,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Оставить только host-строки для обучения legacy baseline."""
    validated = validate_benchmark_frame(df_benchmark, sources=sources)
    host_df = validated[validated[sources.population_col]].reset_index(drop=True)
    if host_df.empty:
        raise ValueError("Legacy Gaussian baseline received no host rows for training.")
    return host_df


def fit_legacy_gaussian_baseline(
    df_train: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    use_m_subclasses: bool = False,
    shrink_alpha: float = 0.15,
) -> LegacyGaussianModel:
    """Обучить legacy baseline на host-части benchmark train split.

    По умолчанию baseline держится максимально простым и использует
    `use_m_subclasses=False`. При необходимости диагностический режим
    с M-подклассами можно включить отдельным параметром.
    """
    host_train_df = select_host_training_rows(df_train, sources=sources)
    return fit_gaussian_model(
        host_train_df,
        use_m_subclasses=use_m_subclasses,
        shrink_alpha=shrink_alpha,
    )


def build_legacy_gaussian_param_grid() -> dict[str, list[object]]:
    """Собрать компактный search-space для legacy Gaussian baseline."""
    return {
        "use_m_subclasses": [False, True],
        "shrink_alpha": [0.05, 0.15, 0.30],
    }


def fit_legacy_gaussian_baseline_with_search(
    df_train: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[LegacyGaussianModel, ModelSearchSummary]:
    """Подобрать и обучить лучший legacy Gaussian baseline по train split."""
    validated = validate_benchmark_frame(df_train, sources=sources)
    return run_manual_model_search(
        validated,
        model_name=LEGACY_GAUSSIAN_MODEL_NAME,
        param_grid=build_legacy_gaussian_param_grid(),
        fit_model=lambda candidate_df, params: fit_legacy_gaussian_baseline(
            candidate_df,
            sources=sources,
            use_m_subclasses=bool(params["use_m_subclasses"]),
            shrink_alpha=float(cast(float, params["shrink_alpha"])),
        ),
        score_model=lambda candidate_model, candidate_df: score_legacy_gaussian_baseline(
            candidate_model,
            candidate_df,
            sources=sources,
        ),
        sources=sources,
        cv_config=cv_config,
        search_config=search_config,
    )


def score_legacy_gaussian_baseline(
    model: LegacyGaussianModel,
    df_part: pd.DataFrame,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
) -> pd.DataFrame:
    """Скорить benchmark frame через legacy Gaussian и вернуть общий contract."""
    validated = validate_benchmark_frame(df_part, sources=sources)
    scored = score_df(
        model=model,
        df=validated,
        spec_class_col=sources.class_col,
    ).copy()
    scored["model_name"] = LEGACY_GAUSSIAN_MODEL_NAME
    scored["model_score"] = scored["similarity"].astype(float)
    return scored


def run_legacy_gaussian_baseline(
    split: BenchmarkSplit,
    *,
    sources: BenchmarkSources = DEFAULT_BENCHMARK_SOURCES,
    cv_config: CrossValidationConfig = DEFAULT_CV_CONFIG,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> LegacyGaussianBaselineRun:
    """Обучить и проскорить legacy baseline на общем benchmark split."""
    model, search_summary = fit_legacy_gaussian_baseline_with_search(
        split.train_df,
        sources=sources,
        cv_config=cv_config,
        search_config=search_config,
    )
    scored_split = ModelScoreFrames(
        model_name=LEGACY_GAUSSIAN_MODEL_NAME,
        train_scored_df=score_legacy_gaussian_baseline(
            model=model,
            df_part=split.train_df,
            sources=sources,
        ),
        test_scored_df=score_legacy_gaussian_baseline(
            model=model,
            df_part=split.test_df,
            sources=sources,
        ),
    )
    return LegacyGaussianBaselineRun(
        model=model,
        search_summary=search_summary,
        scored_split=scored_split,
    )


__all__ = [
    "LEGACY_GAUSSIAN_MODEL_NAME",
    "LegacyGaussianBaselineRun",
    "build_legacy_gaussian_param_grid",
    "fit_legacy_gaussian_baseline",
    "fit_legacy_gaussian_baseline_with_search",
    "run_legacy_gaussian_baseline",
    "score_legacy_gaussian_baseline",
    "select_host_training_rows",
    "validate_benchmark_frame",
]
