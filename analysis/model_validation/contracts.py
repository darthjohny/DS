"""Контракты heavy validation слоя для generalization-аудита.

Модуль фиксирует:

- режимы heavy validation run;
- repeated-split конфигурацию;
- run request и output layout;
- typed result каркаса validation-контура.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from analysis.model_comparison.contracts import ComparisonProtocol

type ValidationMode = Literal["fast", "full"]

VALIDATION_MODES: tuple[ValidationMode, ...] = ("fast", "full")


@dataclass(frozen=True, slots=True)
class RepeatedSplitConfig:
    """Конфигурация repeated split evaluation для heavy validation."""

    random_states: tuple[int, ...] = (11, 17, 23)

    def __post_init__(self) -> None:
        """Проверить, что repeated split контур задан корректно."""
        if not self.random_states:
            raise ValueError("RepeatedSplitConfig.random_states must not be empty.")
        if len(set(self.random_states)) != len(self.random_states):
            raise ValueError(
                "RepeatedSplitConfig.random_states must contain unique values."
            )


@dataclass(frozen=True, slots=True)
class ModelValidationProtocol:
    """Канонический protocol heavy validation слоя."""

    name: str = "model_generalization_validation_v1"
    comparison_protocol: ComparisonProtocol = field(default_factory=ComparisonProtocol)
    repeated_split: RepeatedSplitConfig = field(default_factory=RepeatedSplitConfig)
    primary_metrics: tuple[str, ...] = (
        "roc_auc",
        "pr_auc",
        "brier",
        "precision_at_k",
    )


@dataclass(frozen=True, slots=True)
class ModelValidationRunRequest:
    """Запрос на инициализацию heavy validation run."""

    run_name: str
    output_dir: Path
    mode: ValidationMode = "fast"
    note: str = ""
    include_optional_diagnostics: bool = False

    def __post_init__(self) -> None:
        """Проверить корректность run request."""
        if not self.run_name.strip():
            raise ValueError("ModelValidationRunRequest.run_name must not be empty.")
        if self.mode not in VALIDATION_MODES:
            supported = ", ".join(VALIDATION_MODES)
            raise ValueError(
                "ModelValidationRunRequest.mode must be one of: "
                f"{supported}."
            )


@dataclass(frozen=True, slots=True)
class ModelValidationArtifactPaths:
    """Набор canonical output paths для heavy validation артефактов."""

    output_dir: Path
    report_markdown_path: Path
    repeated_splits_csv_path: Path
    model_summary_csv_path: Path
    generalization_summary_csv_path: Path
    gap_diagnostics_csv_path: Path
    risk_audit_csv_path: Path
    plots_dir: Path


@dataclass(frozen=True, slots=True)
class ModelValidationLayoutResult:
    """Результат инициализации output layout heavy validation run."""

    request: ModelValidationRunRequest
    protocol: ModelValidationProtocol
    artifact_paths: ModelValidationArtifactPaths


@dataclass(slots=True)
class ModelValidationSplitResult:
    """Результат одного repeated split запуска heavy validation."""

    split_random_state: int
    summary_df: pd.DataFrame
    search_summary_df: pd.DataFrame
    generalization_df: pd.DataFrame


@dataclass(slots=True)
class RepeatedSplitEvaluationResult:
    """Агрегированный результат repeated split evaluation."""

    split_results: tuple[ModelValidationSplitResult, ...]
    repeated_splits_df: pd.DataFrame
    model_summary_df: pd.DataFrame
    generalization_summary_df: pd.DataFrame
    gap_diagnostics_df: pd.DataFrame
    risk_audit_df: pd.DataFrame


__all__ = [
    "ModelValidationArtifactPaths",
    "ModelValidationLayoutResult",
    "ModelValidationProtocol",
    "ModelValidationRunRequest",
    "ModelValidationSplitResult",
    "RepeatedSplitEvaluationResult",
    "RepeatedSplitConfig",
    "VALIDATION_MODES",
    "ValidationMode",
]
