"""Публичный API пакета host-модели.

Пакет объединяет:

- контракты и валидацию артефактов;
- загрузку обучающих данных из БД;
- обучение legacy и contrastive моделей;
- scoring для legacy и `host-vs-field` веток;
- общие математические helpers и CLI-точки входа.
"""

from __future__ import annotations

from host_model.artifacts import (
    ContrastiveClassParams,
    ContrastiveGaussianModel,
    ContrastiveModelMeta,
    ContrastivePopulationParams,
    ContrastiveScoreResult,
    HostModelArtifact,
    LegacyClassParams,
    LegacyGaussianModel,
    LegacyModelMeta,
    ScoreResult,
    is_contrastive_model,
    load_contrastive_model,
    load_model,
    require_contrastive_scoring_model,
    require_legacy_scoring_model,
    save_model,
    validate_host_model_artifact,
)
from host_model.cli import main, parse_args
from host_model.constants import (
    CONTRASTIVE_POPULATION_COLUMN,
    CONTRASTIVE_VIEW_ENV,
    DEFAULT_CONTRASTIVE_FIELD_VIEW,
    DEFAULT_CONTRASTIVE_HOST_VIEW,
    DWARF_CLASSES,
    EPS,
    FEATURES,
    LOGG_DWARF_MIN,
    M_EARLY_MAX,
    M_EARLY_MIN,
    M_LATE_MAX,
    M_MID_MAX,
    M_MID_MIN,
)
from host_model.contrastive_score import (
    resolve_contrastive_label,
    score_df_contrastive,
    score_one_contrastive,
)
from host_model.db import (
    load_contrastive_training_from_db,
    load_default_contrastive_training_from_db,
    load_dwarfs_from_db,
    make_engine_from_env,
    resolve_contrastive_view_name,
)
from host_model.fit import (
    build_contrastive_subsets,
    fit_contrastive_gaussian_model,
    fit_gaussian_model,
    fit_population_gaussian,
    split_m_subclasses,
)
from host_model.gaussian_math import (
    choose_m_subclass_label,
    contrastive_host_posterior,
    cov_sample,
    gaussian_log_likelihood,
    has_missing_values,
    is_missing_scalar,
    mahalanobis_distance,
    population_log_likelihood,
    shrink_covariance,
    similarity_from_distance,
    stabilize_covariance,
    zscore_apply,
    zscore_fit,
)
from host_model.legacy_score import score_df, score_df_all_classes, score_one, score_one_all_classes
from host_model.training_data import (
    normalize_host_flag,
    prepare_contrastive_training_df,
)

__all__ = [
    "CONTRASTIVE_POPULATION_COLUMN",
    "CONTRASTIVE_VIEW_ENV",
    "DEFAULT_CONTRASTIVE_FIELD_VIEW",
    "DEFAULT_CONTRASTIVE_HOST_VIEW",
    "DWARF_CLASSES",
    "EPS",
    "FEATURES",
    "LOGG_DWARF_MIN",
    "M_EARLY_MAX",
    "M_EARLY_MIN",
    "M_LATE_MAX",
    "M_MID_MAX",
    "M_MID_MIN",
    "ContrastiveClassParams",
    "ContrastiveGaussianModel",
    "ContrastiveModelMeta",
    "ContrastivePopulationParams",
    "ContrastiveScoreResult",
    "HostModelArtifact",
    "LegacyClassParams",
    "LegacyGaussianModel",
    "LegacyModelMeta",
    "ScoreResult",
    "build_contrastive_subsets",
    "choose_m_subclass_label",
    "contrastive_host_posterior",
    "cov_sample",
    "fit_contrastive_gaussian_model",
    "fit_gaussian_model",
    "fit_population_gaussian",
    "gaussian_log_likelihood",
    "has_missing_values",
    "is_contrastive_model",
    "is_missing_scalar",
    "load_contrastive_model",
    "load_contrastive_training_from_db",
    "load_default_contrastive_training_from_db",
    "load_dwarfs_from_db",
    "load_model",
    "mahalanobis_distance",
    "main",
    "make_engine_from_env",
    "normalize_host_flag",
    "parse_args",
    "population_log_likelihood",
    "prepare_contrastive_training_df",
    "require_contrastive_scoring_model",
    "require_legacy_scoring_model",
    "resolve_contrastive_label",
    "resolve_contrastive_view_name",
    "save_model",
    "score_df",
    "score_df_all_classes",
    "score_df_contrastive",
    "score_one",
    "score_one_all_classes",
    "score_one_contrastive",
    "shrink_covariance",
    "similarity_from_distance",
    "split_m_subclasses",
    "stabilize_covariance",
    "validate_host_model_artifact",
    "zscore_apply",
    "zscore_fit",
]
