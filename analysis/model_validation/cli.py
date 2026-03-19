"""CLI heavy validation слоя."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from analysis.model_validation.contracts import (
    VALIDATION_MODES,
    ModelValidationLayoutResult,
    ModelValidationProtocol,
    ModelValidationRunRequest,
    RepeatedSplitConfig,
    ValidationMode,
)
from analysis.model_validation.layout import (
    DEFAULT_MODEL_VALIDATION_OUTPUT_DIR,
    initialize_model_validation_layout,
)
from analysis.model_validation.repeated_splits import (
    SplitRunner,
    run_repeated_split_evaluation,
)
from analysis.model_validation.reporting import (
    build_validation_layout_markdown,
    save_model_validation_artifacts,
)

FAST_MODE_RANDOM_STATES: tuple[int, ...] = (11, 17, 23)
FULL_MODE_RANDOM_STATES: tuple[int, ...] = (11, 17, 23, 31, 37)


@dataclass(slots=True)
class ModelValidationRunResult:
    """Результат heavy validation run."""

    layout: ModelValidationLayoutResult
    report_markdown_path: Path
    repeated_splits_df: pd.DataFrame | None = None
    model_summary_df: pd.DataFrame | None = None
    generalization_summary_df: pd.DataFrame | None = None
    gap_diagnostics_df: pd.DataFrame | None = None
    risk_audit_df: pd.DataFrame | None = None


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """Разобрать аргументы CLI для heavy validation."""
    parser = ArgumentParser(
        description="Инициализация heavy validation run для generalization-аудита.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Префикс heavy validation артефактов.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_VALIDATION_OUTPUT_DIR,
        help="Каталог сохранения heavy validation артефактов.",
    )
    parser.add_argument(
        "--mode",
        choices=VALIDATION_MODES,
        default="fast",
        help="Режим heavy validation run.",
    )
    parser.add_argument(
        "--random-states",
        type=int,
        nargs="+",
        default=None,
        help="Список random_state для repeated split evaluation.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Произвольное пояснение для markdown scaffold.",
    )
    parser.add_argument(
        "--include-optional-diagnostics",
        action="store_true",
        help="Отметить, что будущий run должен включать optional diagnostics.",
    )
    return parser.parse_args(argv)


def build_protocol_from_args(args: Namespace) -> ModelValidationProtocol:
    """Построить protocol heavy validation из CLI-аргументов."""
    random_states = (
        tuple(int(value) for value in args.random_states)
        if args.random_states is not None
        else (
            FULL_MODE_RANDOM_STATES
            if str(args.mode) == "full"
            else FAST_MODE_RANDOM_STATES
        )
    )
    return ModelValidationProtocol(
        repeated_split=RepeatedSplitConfig(random_states=random_states),
    )


def run_model_validation_scaffold(
    protocol: ModelValidationProtocol,
    *,
    request: ModelValidationRunRequest,
) -> ModelValidationRunResult:
    """Инициализировать output layout heavy validation и записать scaffold."""
    layout = initialize_model_validation_layout(
        request,
        protocol=protocol,
    )
    markdown = build_validation_layout_markdown(layout)
    layout.artifact_paths.report_markdown_path.write_text(
        markdown,
        encoding="utf-8",
    )
    return ModelValidationRunResult(
        layout=layout,
        report_markdown_path=layout.artifact_paths.report_markdown_path,
    )


def run_model_validation(
    protocol: ModelValidationProtocol,
    *,
    request: ModelValidationRunRequest,
    run_split: SplitRunner | None = None,
) -> ModelValidationRunResult:
    """Запустить repeated split heavy validation и сохранить артефакты."""
    layout = initialize_model_validation_layout(
        request,
        protocol=protocol,
    )
    evaluation = run_repeated_split_evaluation(protocol, run_split=run_split)
    save_model_validation_artifacts(
        layout,
        repeated_splits_df=evaluation.repeated_splits_df,
        model_summary_df=evaluation.model_summary_df,
        generalization_summary_df=evaluation.generalization_summary_df,
        gap_diagnostics_df=evaluation.gap_diagnostics_df,
        risk_audit_df=evaluation.risk_audit_df,
    )
    return ModelValidationRunResult(
        layout=layout,
        report_markdown_path=layout.artifact_paths.report_markdown_path,
        repeated_splits_df=evaluation.repeated_splits_df,
        model_summary_df=evaluation.model_summary_df,
        generalization_summary_df=evaluation.generalization_summary_df,
        gap_diagnostics_df=evaluation.gap_diagnostics_df,
        risk_audit_df=evaluation.risk_audit_df,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить heavy validation run."""
    args = parse_args(argv)
    protocol = build_protocol_from_args(args)
    request = ModelValidationRunRequest(
        run_name=str(args.run_name),
        output_dir=Path(args.output_dir),
        mode=cast(ValidationMode, str(args.mode)),
        note=str(args.note),
        include_optional_diagnostics=bool(args.include_optional_diagnostics),
    )
    result = run_model_validation(
        protocol,
        request=request,
    )
    print(f"Heavy validation completed: {result.report_markdown_path}")
    return 0


__all__ = [
    "FAST_MODE_RANDOM_STATES",
    "FULL_MODE_RANDOM_STATES",
    "ModelValidationRunResult",
    "build_protocol_from_args",
    "main",
    "parse_args",
    "run_model_validation",
    "run_model_validation_scaffold",
]
