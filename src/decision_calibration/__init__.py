"""Публичный API пакета офлайн-калибровки decision layer.

Пакет объединяет:

- конфигурацию calibration factors;
- runtime-загрузку validated dataset и base scoring;
- применение офлайн-формулы decision layer;
- сборку markdown- и CSV-артефактов итерации;
- CLI-точку входа калибратора.
"""

from __future__ import annotations

from decision_calibration.cli import main
from decision_calibration.config import (
    CalibrationConfig,
    ClassPriorConfig,
    DistanceConfig,
    MetallicityConfig,
    ParallaxPrecisionConfig,
    QualityConfig,
    RuweConfig,
    load_calibration_config,
    normalize_json_object,
    parse_args,
)
from decision_calibration.constants import CALIBRATOR_VERSION, DEFAULT_TOP_N
from decision_calibration.reporting import (
    IterationSummary,
    build_iteration_markdown,
    build_iteration_summary,
    class_distribution_frame,
    frame_to_text,
    print_summary,
    save_iteration_artifacts,
    score_summary_frame,
    top_candidates_frame,
)
from decision_calibration.runtime import (
    BaseScoringResult,
    ReadyDatasetRecord,
    fetch_ready_dataset_record,
    load_ready_input_dataset,
    make_run_id,
    run_base_scoring,
)
from decision_calibration.scoring import (
    apply_calibration_config,
    build_low_priority_preview,
    build_unknown_preview,
    class_prior,
    distance_factor,
    distance_pc_from_parallax,
    metallicity_factor,
    parallax_precision_factor,
    quality_factor,
    ruwe_factor,
)

__all__ = [
    "BaseScoringResult",
    "CALIBRATOR_VERSION",
    "CalibrationConfig",
    "ClassPriorConfig",
    "DEFAULT_TOP_N",
    "DistanceConfig",
    "IterationSummary",
    "MetallicityConfig",
    "ParallaxPrecisionConfig",
    "QualityConfig",
    "ReadyDatasetRecord",
    "RuweConfig",
    "apply_calibration_config",
    "build_iteration_markdown",
    "build_iteration_summary",
    "build_low_priority_preview",
    "build_unknown_preview",
    "class_distribution_frame",
    "class_prior",
    "distance_factor",
    "distance_pc_from_parallax",
    "fetch_ready_dataset_record",
    "frame_to_text",
    "load_calibration_config",
    "load_ready_input_dataset",
    "main",
    "make_run_id",
    "metallicity_factor",
    "normalize_json_object",
    "parallax_precision_factor",
    "parse_args",
    "print_summary",
    "quality_factor",
    "run_base_scoring",
    "ruwe_factor",
    "save_iteration_artifacts",
    "score_summary_frame",
    "top_candidates_frame",
]
