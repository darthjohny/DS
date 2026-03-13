"""Контракты comparison-layer для baseline-моделей.

Модуль фиксирует:

- источники benchmark dataset;
- параметры train/test split;
- параметры cross-validation и model search;
- общий protocol первой волны baseline-сравнения.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from host_model.constants import (
    CONTRASTIVE_POPULATION_COLUMN,
    DEFAULT_CONTRASTIVE_FIELD_VIEW,
    DEFAULT_CONTRASTIVE_HOST_VIEW,
    DWARF_CLASSES,
    FEATURES,
)

type RandomForestClassWeight = Literal["balanced", "balanced_subsample"] | None
type MLPActivation = Literal["identity", "logistic", "tanh", "relu"]
type MLPSolver = Literal["lbfgs", "sgd", "adam"]
type SearchRefitMetric = Literal["roc_auc", "pr_auc", "brier", "precision_at_k"]

SEARCH_REFIT_METRICS: tuple[SearchRefitMetric, ...] = (
    "roc_auc",
    "pr_auc",
    "brier",
    "precision_at_k",
)


@dataclass(frozen=True, slots=True)
class BenchmarkSources:
    """Источники benchmark dataset для задачи `host vs field`.

    Источник `host` берётся из host-train relation, а источник `field`
    из Gaia reference dwarf-population. Оба relation должны содержать
    `source_id`, `spec_class` и признаки из `feature_columns`.
    """

    host_view: str = DEFAULT_CONTRASTIVE_HOST_VIEW
    field_view: str = DEFAULT_CONTRASTIVE_FIELD_VIEW
    source_id_col: str = "source_id"
    class_col: str = "spec_class"
    population_col: str = CONTRASTIVE_POPULATION_COLUMN
    feature_columns: tuple[str, ...] = tuple(FEATURES)
    allowed_classes: tuple[str, ...] = tuple(DWARF_CLASSES)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Параметры детерминированного train/test split для benchmark."""

    test_size: float = 0.30
    random_state: int = 42

    def __post_init__(self) -> None:
        """Проверить базовую корректность split-настроек."""
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("SplitConfig.test_size must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class CrossValidationConfig:
    """Параметры cross-validation для benchmark tuning-контура."""

    n_splits: int = 10
    shuffle: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        """Проверить базовую корректность CV-настроек."""
        if self.n_splits < 2:
            raise ValueError("CrossValidationConfig.n_splits must be at least 2.")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Параметры выбора гиперпараметров для comparison-layer."""

    refit_metric: SearchRefitMetric = "roc_auc"
    precision_k: int = 50

    def __post_init__(self) -> None:
        """Проверить, что search-контракт задан корректно."""
        if self.refit_metric not in SEARCH_REFIT_METRICS:
            supported = ", ".join(SEARCH_REFIT_METRICS)
            raise ValueError(
                "SearchConfig.refit_metric must be one of: "
                f"{supported}."
            )
        if self.precision_k <= 0:
            raise ValueError("SearchConfig.precision_k must be greater than 0.")


@dataclass(frozen=True, slots=True)
class ComparisonProtocol:
    """Общий protocol baseline-сравнения comparison-layer."""

    name: str = "baseline_host_vs_field_v1"
    sources: BenchmarkSources = field(default_factory=BenchmarkSources)
    split: SplitConfig = field(default_factory=SplitConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    primary_metrics: tuple[str, ...] = (
        "roc_auc",
        "pr_auc",
        "brier",
        "precision_at_k",
    )
    snapshot_relation: str = "public.gaia_dr3_training"


@dataclass(slots=True)
class ModelScoreFrames:
    """Пара scored DataFrame для train/test частей одной модели."""

    model_name: str
    train_scored_df: pd.DataFrame
    test_scored_df: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ClassSearchSummary:
    """Сводка выбора гиперпараметров для одного class-specific model head."""

    model_name: str
    spec_class: str
    refit_metric: SearchRefitMetric
    precision_k: int
    cv_folds: int
    n_train_rows: int
    n_host: int
    n_field: int
    candidate_count: int
    best_cv_score: float
    best_params: dict[str, object]


@dataclass(frozen=True, slots=True)
class ModelSearchSummary:
    """Сводка выбора гиперпараметров для модели на всём train split."""

    model_name: str
    refit_metric: SearchRefitMetric
    precision_k: int
    cv_folds: int
    n_train_rows: int
    n_host: int
    n_field: int
    candidate_count: int
    best_cv_score: float
    best_params: dict[str, object]


@dataclass(frozen=True, slots=True)
class RandomForestConfig:
    """Параметры class-specific RandomForest baseline.

    Первая волна держит baseline намеренно простым и воспроизводимым:

    - одна модель на каждый `spec_class`;
    - фиксированный `random_state`;
    - `class_weight='balanced'`;
    - `n_jobs=1`, чтобы исключить лишние вариации runtime.
    """

    n_estimators: int = 300
    min_samples_leaf: int = 2
    random_state: int = 42
    class_weight: RandomForestClassWeight = "balanced"
    n_jobs: int = 1


@dataclass(frozen=True, slots=True)
class MLPBaselineConfig:
    """Параметры компактного class-specific MLP baseline.

    Контракт первой волны намеренно держится простым:

    - используется только на задаче `host vs field` внутри `M/K/G/F`;
    - признаки совпадают с базовым physical feature set benchmark;
    - обязательная нормализация признаков выполняется в wrapper-е через
      `StandardScaler`;
    - сеть остаётся маленькой, чтобы не раздувать сложность и риск
      переобучения.
    """

    hidden_layer_sizes: tuple[int, ...] = (16, 8)
    activation: MLPActivation = "relu"
    solver: MLPSolver = "adam"
    alpha: float = 0.001
    learning_rate_init: float = 0.001
    batch_size: int = 64
    max_iter: int = 500
    tol: float = 0.001
    early_stopping: bool = True
    validation_fraction: float = 0.10
    n_iter_no_change: int = 20
    random_state: int = 42


DEFAULT_BENCHMARK_SOURCES = BenchmarkSources()
DEFAULT_CV_CONFIG = CrossValidationConfig()
DEFAULT_SPLIT_CONFIG = SplitConfig()
DEFAULT_COMPARISON_PROTOCOL = ComparisonProtocol()
DEFAULT_MLP_BASELINE_CONFIG = MLPBaselineConfig()
DEFAULT_RANDOM_FOREST_CONFIG = RandomForestConfig()
DEFAULT_SEARCH_CONFIG = SearchConfig()

__all__ = [
    "BenchmarkSources",
    "ClassSearchSummary",
    "ComparisonProtocol",
    "CrossValidationConfig",
    "DEFAULT_BENCHMARK_SOURCES",
    "DEFAULT_COMPARISON_PROTOCOL",
    "DEFAULT_CV_CONFIG",
    "DEFAULT_MLP_BASELINE_CONFIG",
    "DEFAULT_RANDOM_FOREST_CONFIG",
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_SPLIT_CONFIG",
    "MLPActivation",
    "MLPBaselineConfig",
    "ModelSearchSummary",
    "ModelScoreFrames",
    "MLPSolver",
    "RandomForestClassWeight",
    "RandomForestConfig",
    "SEARCH_REFIT_METRICS",
    "SearchConfig",
    "SearchRefitMetric",
    "SplitConfig",
]
