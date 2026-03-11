"""Фасад совместимости для офлайн-калибровки decision layer.

Что делает модуль:
    - реэкспортирует публичный API пакета `decision_calibration`;
    - сохраняет старую точку входа `decision_layer_calibrator.py`;
    - оставляет прежний импортный путь для config, runtime, scoring,
      reporting и CLI-функций калибровки.

Где находится основная логика:
    - конфигурация: `decision_calibration.config`;
    - загрузка данных и базовый scoring: `decision_calibration.runtime`;
    - применение calibration factors: `decision_calibration.scoring`;
    - сборка markdown и CSV артефактов: `decision_calibration.reporting`.

Что модуль не делает:
    - не содержит собственной calibration-логики;
    - нужен прежде всего для обратной совместимости после рефакторинга.
"""

from __future__ import annotations

from decision_calibration import (
    CALIBRATOR_VERSION,
    DEFAULT_TOP_N,
    BaseScoringResult,
    CalibrationConfig,
    ClassPriorConfig,
    DistanceConfig,
    IterationSummary,
    MetallicityConfig,
    ParallaxPrecisionConfig,
    QualityConfig,
    ReadyDatasetRecord,
    RuweConfig,
    apply_calibration_config,
    build_iteration_markdown,
    build_iteration_summary,
    build_low_priority_preview,
    class_distribution_frame,
    class_prior,
    distance_factor,
    distance_pc_from_parallax,
    fetch_ready_dataset_record,
    frame_to_text,
    load_calibration_config,
    load_ready_input_dataset,
    main,
    make_run_id,
    metallicity_factor,
    normalize_json_object,
    parallax_precision_factor,
    parse_args,
    print_summary,
    quality_factor,
    run_base_scoring,
    ruwe_factor,
    save_iteration_artifacts,
    score_summary_frame,
    top_candidates_frame,
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


if __name__ == "__main__":
    main()
