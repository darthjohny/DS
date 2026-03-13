"""Публичный API пакета физического Gaussian router.

Пакет публикует несколько групп API:

- контракты артефактов и JSON IO;
- загрузку обучающей выборки из БД;
- label- и math-helpers для router-модели;
- обучение, scoring и CLI-точку входа.
"""

from router_model.artifacts import (
    DEFAULT_OOD_POLICY_VERSION,
    DISABLED_OOD_POLICY_VERSION,
    RouterClassParams,
    RouterMeta,
    RouterModel,
    RouterOODMeta,
    RouterScoreResult,
    build_router_meta,
    load_router_model,
    normalize_router_meta,
    normalize_router_model,
    save_router_model,
)
from router_model.cli import main
from router_model.db import (
    FEATURES,
    ROUTER_VIEW,
    load_router_training_from_db,
    make_engine_from_env,
)
from router_model.fit import ROUTER_MODEL_VERSION, fit_router_model
from router_model.labels import (
    EVOLUTION_STAGES,
    SPEC_CLASSES,
    has_missing_values,
    is_missing_scalar,
    make_router_label,
    normalize_evolution_stage,
    normalize_spec_class,
    split_router_label,
)
from router_model.math import (
    cov_sample,
    mahalanobis_distance,
    router_log_likelihood,
    shrink_covariance,
    similarity_from_distance,
    stabilize_covariance,
    uniform_log_prior,
    zscore_apply,
    zscore_fit,
)
from router_model.ood import (
    RouterOODDecision,
    RouterOODPolicy,
    apply_ood_policy,
    build_unknown_router_score,
    evaluate_router_ood,
    load_ood_policy,
)
from router_model.score import (
    build_raw_router_score,
    rank_router_candidates,
    score_router_df,
    score_router_one,
)

__all__ = [
    "RouterClassParams",
    "RouterMeta",
    "RouterModel",
    "RouterOODMeta",
    "RouterScoreResult",
    "DEFAULT_OOD_POLICY_VERSION",
    "DISABLED_OOD_POLICY_VERSION",
    "FEATURES",
    "SPEC_CLASSES",
    "EVOLUTION_STAGES",
    "ROUTER_VIEW",
    "ROUTER_MODEL_VERSION",
    "make_engine_from_env",
    "load_router_training_from_db",
    "zscore_fit",
    "zscore_apply",
    "cov_sample",
    "shrink_covariance",
    "mahalanobis_distance",
    "similarity_from_distance",
    "stabilize_covariance",
    "router_log_likelihood",
    "uniform_log_prior",
    "is_missing_scalar",
    "has_missing_values",
    "normalize_spec_class",
    "normalize_evolution_stage",
    "make_router_label",
    "split_router_label",
    "RouterOODDecision",
    "RouterOODPolicy",
    "load_ood_policy",
    "evaluate_router_ood",
    "apply_ood_policy",
    "build_unknown_router_score",
    "fit_router_model",
    "build_router_meta",
    "normalize_router_meta",
    "normalize_router_model",
    "rank_router_candidates",
    "build_raw_router_score",
    "score_router_one",
    "score_router_df",
    "save_router_model",
    "load_router_model",
    "main",
]
