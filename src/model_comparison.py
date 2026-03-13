"""Фасад совместимости для comparative benchmark baseline-слоя.

Что делает модуль:
    - реэкспортирует CLI и публичные helper-ы пакета
      `analysis.model_comparison`;
    - сохраняет простую точку входа `python src/model_comparison.py`;
    - не содержит собственной логики обучения, snapshot или метрик.

Где находится основная логика:
    - protocol, split и контракты: `analysis.model_comparison.contracts`,
      `analysis.model_comparison.data`;
    - wrappers моделей: `analysis.model_comparison.contrastive`,
      `analysis.model_comparison.legacy_gaussian`,
      `analysis.model_comparison.random_forest`;
    - метрики, report и snapshot:
      `analysis.model_comparison.metrics`,
      `analysis.model_comparison.reporting`,
      `analysis.model_comparison.snapshot`;
    - CLI orchestration: `analysis.model_comparison.cli`.

Что модуль не делает:
    - не вмешивается в production pipeline;
    - не пишет данные в боевые result-таблицы;
    - нужен в первую очередь как удобная фасадная точка входа.
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.model_comparison import (
    DEFAULT_MLP_BASELINE_CONFIG,
    LEGACY_GAUSSIAN_MODEL_NAME,
    MAIN_CONTRASTIVE_MODEL_NAME,
    MLP_BASELINE_MODEL_NAME,
    RANDOM_FOREST_MODEL_NAME,
    BenchmarkSources,
    BenchmarkSplit,
    ComparisonProtocol,
    ContrastiveModelRun,
    LegacyGaussianBaselineRun,
    MLPBaselineConfig,
    MLPBaselineRun,
    ModelScoreFrames,
    RandomForestBaselineRun,
    RandomForestConfig,
    SnapshotComparisonResult,
    SnapshotModelRun,
    SplitConfig,
    build_classwise_metrics_frame,
    build_comparison_classwise_frame,
    build_comparison_markdown,
    build_comparison_summary_frame,
    build_metrics_frame,
    build_snapshot_markdown,
    build_snapshot_summary_frame,
    build_snapshot_top_frame,
    build_stratify_labels,
    fit_legacy_gaussian_baseline,
    fit_main_contrastive_model,
    fit_mlp_baseline,
    fit_random_forest_baseline,
    load_and_split_benchmark_dataset,
    load_benchmark_dataset,
    precision_at_k,
    prepare_benchmark_dataset,
    run_legacy_gaussian_baseline,
    run_main_contrastive_model,
    run_mlp_baseline,
    run_random_forest_baseline,
    run_snapshot_comparison,
    save_comparison_artifacts,
    save_snapshot_artifacts,
    score_legacy_gaussian_baseline,
    score_main_contrastive_model,
    score_mlp_baseline,
    score_random_forest_baseline,
    split_benchmark_dataset,
    validate_scored_frame,
)
from analysis.model_comparison.cli import (
    ComparisonRunResult,
    build_protocol_from_args,
    default_run_name,
    main,
    parse_args,
    print_summary,
    run_model_comparison,
)

__all__ = [
    "BenchmarkSources",
    "BenchmarkSplit",
    "ComparisonProtocol",
    "ComparisonRunResult",
    "ContrastiveModelRun",
    "DEFAULT_MLP_BASELINE_CONFIG",
    "LEGACY_GAUSSIAN_MODEL_NAME",
    "LegacyGaussianBaselineRun",
    "MAIN_CONTRASTIVE_MODEL_NAME",
    "MLP_BASELINE_MODEL_NAME",
    "MLPBaselineConfig",
    "MLPBaselineRun",
    "ModelScoreFrames",
    "RANDOM_FOREST_MODEL_NAME",
    "RandomForestBaselineRun",
    "RandomForestConfig",
    "SnapshotComparisonResult",
    "SnapshotModelRun",
    "SplitConfig",
    "build_classwise_metrics_frame",
    "build_comparison_classwise_frame",
    "build_comparison_markdown",
    "build_comparison_summary_frame",
    "build_metrics_frame",
    "build_snapshot_markdown",
    "build_snapshot_summary_frame",
    "build_snapshot_top_frame",
    "build_protocol_from_args",
    "build_stratify_labels",
    "default_run_name",
    "fit_legacy_gaussian_baseline",
    "fit_mlp_baseline",
    "fit_main_contrastive_model",
    "fit_random_forest_baseline",
    "load_and_split_benchmark_dataset",
    "load_benchmark_dataset",
    "main",
    "parse_args",
    "precision_at_k",
    "prepare_benchmark_dataset",
    "print_summary",
    "run_legacy_gaussian_baseline",
    "run_mlp_baseline",
    "run_main_contrastive_model",
    "run_model_comparison",
    "run_random_forest_baseline",
    "run_snapshot_comparison",
    "save_comparison_artifacts",
    "save_snapshot_artifacts",
    "score_legacy_gaussian_baseline",
    "score_main_contrastive_model",
    "score_mlp_baseline",
    "score_random_forest_baseline",
    "split_benchmark_dataset",
    "validate_scored_frame",
]


if __name__ == "__main__":
    main()
