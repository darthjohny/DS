"""Публичный API production priority pipeline.

Пакет публикует:

- константы и типизированный контракт результата;
- загрузку входных данных и моделей;
- decision-layer helpers и ветвление пайплайна;
- orchestration, preview и persist результатов в БД.
"""

from __future__ import annotations

from priority_pipeline.branching import (
    RouterBranchFrames,
    is_host_candidate,
    is_unknown_router_output,
    known_low_reason_code,
    split_branches,
    split_router_branches,
)
from priority_pipeline.constants import (
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
)
from priority_pipeline.contracts import PipelineRunResult
from priority_pipeline.decision import (
    apply_common_factors,
    build_low_priority_stub,
    build_unknown_priority_stub,
    class_prior,
    clip_unit_interval,
    color_factor,
    distance_factor,
    host_model_version,
    iter_triplets,
    metallicity_factor,
    normalized_validation_factor,
    order_priority_results,
    parallax_precision_factor,
    priority_tier_from_score,
    quality_factor,
    run_host_similarity,
    ruwe_factor,
)
from priority_pipeline.frame_contract import ensure_decision_columns
from priority_pipeline.input_data import (
    load_input_candidates,
    load_models,
    make_run_id,
)
from priority_pipeline.persist import (
    build_persist_payload,
    save_priority_results,
    save_router_results,
)
from priority_pipeline.pipeline import main, print_preview, run_pipeline, run_router
from priority_pipeline.relations import (
    relation_columns,
    relation_exists,
    split_relation_name,
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
    "run_host_similarity",
    "run_pipeline",
    "run_router",
    "ruwe_factor",
    "save_priority_results",
    "save_router_results",
    "is_host_candidate",
    "is_unknown_router_output",
    "known_low_reason_code",
    "split_branches",
    "split_router_branches",
    "split_relation_name",
]
