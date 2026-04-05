# Файл `model_pipeline_review.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from exohost.reporting.model_pipeline_review_artifacts import (
    build_model_artifact_summary_frame,
    build_threshold_artifact_summary_frame,
)
from exohost.reporting.model_pipeline_review_bundle import (
    load_benchmark_metadata_only,
    load_benchmark_review_bundle,
)
from exohost.reporting.model_pipeline_review_contracts import BenchmarkReviewBundle
from exohost.reporting.model_pipeline_review_stage_frames import (
    build_split_metrics_frame,
    build_target_distribution_frame,
)
from exohost.reporting.model_pipeline_review_summary import (
    build_benchmark_summary_frame,
    build_pipeline_stage_overview_frame,
    build_stage_metric_long_frame,
)

__all__ = [
    "BenchmarkReviewBundle",
    "build_benchmark_summary_frame",
    "build_model_artifact_summary_frame",
    "build_pipeline_stage_overview_frame",
    "build_split_metrics_frame",
    "build_stage_metric_long_frame",
    "build_target_distribution_frame",
    "build_threshold_artifact_summary_frame",
    "load_benchmark_metadata_only",
    "load_benchmark_review_bundle",
]
