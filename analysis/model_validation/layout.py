"""Output layout heavy validation слоя.

Модуль отвечает только за:

- canonical output directory;
- стабильные имена heavy validation артефактов;
- инициализацию каталогов под будущие repeated-split run-ы.
"""

from __future__ import annotations

from pathlib import Path

from analysis.model_validation.contracts import (
    ModelValidationArtifactPaths,
    ModelValidationLayoutResult,
    ModelValidationProtocol,
    ModelValidationRunRequest,
)

DEFAULT_MODEL_VALIDATION_OUTPUT_DIR = Path("experiments/model_validation")


def build_model_validation_artifact_paths(
    request: ModelValidationRunRequest,
) -> ModelValidationArtifactPaths:
    """Собрать stable output paths для одного heavy validation run."""
    run_prefix = request.run_name
    plots_dir = request.output_dir / f"{run_prefix}_plots"
    return ModelValidationArtifactPaths(
        output_dir=request.output_dir,
        report_markdown_path=request.output_dir / f"{run_prefix}_validation_report.md",
        repeated_splits_csv_path=request.output_dir
        / f"{run_prefix}_repeated_splits.csv",
        model_summary_csv_path=request.output_dir / f"{run_prefix}_model_summary.csv",
        generalization_summary_csv_path=request.output_dir
        / f"{run_prefix}_generalization_summary.csv",
        gap_diagnostics_csv_path=request.output_dir
        / f"{run_prefix}_gap_diagnostics.csv",
        risk_audit_csv_path=request.output_dir / f"{run_prefix}_risk_audit.csv",
        plots_dir=plots_dir,
    )


def initialize_model_validation_layout(
    request: ModelValidationRunRequest,
    *,
    protocol: ModelValidationProtocol,
) -> ModelValidationLayoutResult:
    """Создать каталоги и вернуть output layout heavy validation run."""
    request.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = build_model_validation_artifact_paths(request)
    artifact_paths.plots_dir.mkdir(parents=True, exist_ok=True)
    return ModelValidationLayoutResult(
        request=request,
        protocol=protocol,
        artifact_paths=artifact_paths,
    )


__all__ = [
    "DEFAULT_MODEL_VALIDATION_OUTPUT_DIR",
    "build_model_validation_artifact_paths",
    "initialize_model_validation_layout",
]
