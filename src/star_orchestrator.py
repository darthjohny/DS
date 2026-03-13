"""Фасад совместимости для production priority pipeline.

Что делает модуль:
    - реэкспортирует публичный API пакета `priority_pipeline`;
    - сохраняет старую точку входа `star_orchestrator.py`;
    - оставляет совместимый wrapper `run_host_similarity()`
      для старых импортов и тестовых monkeypatch-сценариев.

Где находится основная логика:
    - загрузка входа и моделей: `priority_pipeline.input_data`;
    - decision layer и ветвление: `priority_pipeline.decision`;
    - orchestration и CLI: `priority_pipeline.pipeline`;
    - persist в БД: `priority_pipeline.persist`.

Что модуль не делает:
    - не является каноническим местом развития pipeline;
    - не должен получать новую доменную логику, если её можно
      разместить в `priority_pipeline`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from host_model import score_df_contrastive as score_host_df
from priority_pipeline import (
    DATA_DIR,
    DEFAULT_INPUT_SOURCE,
    DEFAULT_PRIORITY_RESULTS_TABLE,
    DEFAULT_ROUTER_RESULTS_TABLE,
    EVOLVED_STAR_REASON,
    FILTERED_OUT_REASON,
    HOST_MODEL_PATH,
    HOST_MODEL_VERSION,
    HOST_SCORING_REASON,
    HOT_STAR_REASON,
    INPUT_COLUMNS,
    MKGF_CLASSES,
    PRIORITY_REQUIRED_DB_COLUMNS,
    PRIORITY_RESULTS_COLUMNS,
    PROJECT_ROOT,
    ROUTER_MODEL_PATH,
    ROUTER_REQUIRED_DB_COLUMNS,
    ROUTER_RESULTS_COLUMNS,
    ROUTER_UNKNOWN_REASON,
    PipelineRunResult,
    RouterBranchFrames,
    apply_common_factors,
    build_low_priority_stub,
    build_persist_payload,
    build_unknown_priority_stub,
    class_prior,
    clip_unit_interval,
    color_factor,
    distance_factor,
    ensure_decision_columns,
    host_model_version,
    is_host_candidate,
    is_unknown_router_output,
    iter_triplets,
    known_low_reason_code,
    load_input_candidates,
    load_models,
    main,
    make_run_id,
    metallicity_factor,
    normalized_validation_factor,
    order_priority_results,
    parallax_precision_factor,
    print_preview,
    priority_tier_from_score,
    quality_factor,
    relation_columns,
    relation_exists,
    run_pipeline,
    run_router,
    ruwe_factor,
    save_priority_results,
    save_router_results,
    split_branches,
    split_relation_name,
    split_router_branches,
)

__all__ = [
    "DATA_DIR",
    "DEFAULT_INPUT_SOURCE",
    "DEFAULT_PRIORITY_RESULTS_TABLE",
    "DEFAULT_ROUTER_RESULTS_TABLE",
    "EVOLVED_STAR_REASON",
    "FILTERED_OUT_REASON",
    "HOST_MODEL_PATH",
    "HOST_MODEL_VERSION",
    "HOST_SCORING_REASON",
    "HOT_STAR_REASON",
    "INPUT_COLUMNS",
    "MKGF_CLASSES",
    "PRIORITY_REQUIRED_DB_COLUMNS",
    "PRIORITY_RESULTS_COLUMNS",
    "PROJECT_ROOT",
    "PipelineRunResult",
    "ROUTER_UNKNOWN_REASON",
    "ROUTER_MODEL_PATH",
    "ROUTER_REQUIRED_DB_COLUMNS",
    "ROUTER_RESULTS_COLUMNS",
    "RouterBranchFrames",
    "apply_common_factors",
    "build_low_priority_stub",
    "build_unknown_priority_stub",
    "build_persist_payload",
    "class_prior",
    "clip_unit_interval",
    "color_factor",
    "distance_factor",
    "ensure_decision_columns",
    "host_model_version",
    "iter_triplets",
    "load_input_candidates",
    "load_models",
    "main",
    "make_run_id",
    "metallicity_factor",
    "normalized_validation_factor",
    "order_priority_results",
    "parallax_precision_factor",
    "print_preview",
    "priority_tier_from_score",
    "quality_factor",
    "relation_columns",
    "relation_exists",
    "is_host_candidate",
    "is_unknown_router_output",
    "known_low_reason_code",
    "run_host_similarity",
    "run_pipeline",
    "run_router",
    "ruwe_factor",
    "save_priority_results",
    "save_router_results",
    "split_branches",
    "split_router_branches",
    "split_relation_name",
    "score_host_df",
]


def run_host_similarity(
    df_host: pd.DataFrame,
    host_model: Mapping[str, Any],
) -> pd.DataFrame:
    """Совместимый wrapper для host-scoring в старом `star_orchestrator`.

    Функция повторяет контракт production host-ветки, но использует
    локальный alias `score_host_df`, чтобы тесты и старые импорты могли
    подменять scoring через monkeypatch без обращения к пакетной
    реализации в `priority_pipeline.decision`.
    """
    if df_host.empty:
        return df_host.copy()

    scored = score_host_df(
        model=dict(host_model),
        df=df_host,
        spec_class_col="predicted_spec_class",
    )
    scored = apply_common_factors(scored)

    scoring_rows = scored[
        [
            "host_posterior",
            "class_prior",
            "quality_factor",
            "metallicity_factor",
            "color_factor",
            "validation_factor",
        ]
    ].itertuples(index=False, name=None)
    scored["final_score"] = [
        clip_unit_interval(
            float(host_posterior)
            * float(class_prior_value)
            * float(quality_value)
            * float(metallicity_value)
            * float(color_value)
            * float(validation_value)
        )
        for (
            host_posterior,
            class_prior_value,
            quality_value,
            metallicity_value,
            color_value,
            validation_value,
        ) in scoring_rows
    ]
    scored["d_mahal"] = None
    scored["similarity"] = None
    scored["priority_tier"] = [
        priority_tier_from_score(float(score))
        for score in scored["final_score"]
    ]
    scored["reason_code"] = HOST_SCORING_REASON
    scored["host_model_version"] = host_model_version(host_model)
    return scored


if __name__ == "__main__":
    main()
