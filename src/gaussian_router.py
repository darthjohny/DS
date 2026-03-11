"""Фасад совместимости для физического Gaussian router.

Что делает модуль:
    - реэкспортирует публичный API пакета `router_model`;
    - сохраняет совместимость со старыми импортами вида
      `from gaussian_router import ...`;
    - оставляет прежнюю CLI-точку входа через `main()`.

Где находится основная логика:
    - обучение router-модели: `router_model.fit`;
    - scoring и posterior-логика: `router_model.score`;
    - численные примитивы: `router_model.math`;
    - загрузка данных и артефактов: `router_model.db`,
      `router_model.artifacts`.

Что модуль не делает:
    - не содержит собственной доменной логики router;
    - не реализует отдельный production pipeline;
    - не хранит историческое описание старой монолитной версии файла.
"""

from router_model import (
    EVOLUTION_STAGES,
    FEATURES,
    ROUTER_MODEL_VERSION,
    ROUTER_VIEW,
    SPEC_CLASSES,
    RouterClassParams,
    RouterMeta,
    RouterModel,
    RouterScoreResult,
    cov_sample,
    fit_router_model,
    has_missing_values,
    is_missing_scalar,
    load_router_model,
    load_router_training_from_db,
    mahalanobis_distance,
    main,
    make_engine_from_env,
    make_router_label,
    normalize_evolution_stage,
    normalize_spec_class,
    router_log_likelihood,
    save_router_model,
    score_router_df,
    score_router_one,
    shrink_covariance,
    similarity_from_distance,
    split_router_label,
    stabilize_covariance,
    uniform_log_prior,
    zscore_apply,
    zscore_fit,
)

__all__ = [
    "RouterClassParams",
    "RouterMeta",
    "RouterModel",
    "RouterScoreResult",
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
    "fit_router_model",
    "score_router_one",
    "score_router_df",
    "save_router_model",
    "load_router_model",
    "main",
]


if __name__ == "__main__":
    main()
