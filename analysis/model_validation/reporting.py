"""Reporting heavy validation слоя."""

from __future__ import annotations

import pandas as pd

from analysis.model_validation.contracts import ModelValidationLayoutResult


def frame_to_text(df: pd.DataFrame) -> str:
    """Преобразовать DataFrame в компактный текстовый блок для markdown."""
    if df.empty:
        return "Пусто"
    return df.to_string(index=False)


def build_validation_layout_markdown(
    layout: ModelValidationLayoutResult,
) -> str:
    """Собрать markdown scaffold для heavy validation run."""
    request = layout.request
    protocol = layout.protocol
    artifact_paths = layout.artifact_paths
    repeated_states = ", ".join(str(state) for state in protocol.repeated_split.random_states)

    note_block = ""
    if request.note:
        note_block = f"\n## Note\n\n{request.note}\n"

    return "\n".join(
        [
            f"# Model Validation Run: `{request.run_name}`",
            "",
            "## Status",
            "",
            "- stage: `layout_initialized`",
            f"- mode: `{request.mode}`",
            f"- include_optional_diagnostics: `{request.include_optional_diagnostics}`",
            "",
            "## Protocol",
            "",
            f"- validation protocol: `{protocol.name}`",
            f"- benchmark protocol: `{protocol.comparison_protocol.name}`",
            f"- repeated split random states: `{repeated_states}`",
            f"- primary metrics: `{', '.join(protocol.primary_metrics)}`",
            note_block.rstrip(),
            "",
            "## Planned Artifacts",
            "",
            f"- report: `{artifact_paths.report_markdown_path.name}`",
            f"- repeated splits: `{artifact_paths.repeated_splits_csv_path.name}`",
            f"- model summary: `{artifact_paths.model_summary_csv_path.name}`",
            f"- generalization summary: `{artifact_paths.generalization_summary_csv_path.name}`",
            f"- gap diagnostics: `{artifact_paths.gap_diagnostics_csv_path.name}`",
            f"- risk audit: `{artifact_paths.risk_audit_csv_path.name}`",
            f"- plots dir: `{artifact_paths.plots_dir.name}`",
            "",
            "## Scope",
            "",
            "- Этот scaffold фиксирует contracts и artifact layout heavy validation.",
            "- Repeated split evaluation, stage summary и gap diagnostics добавляются следующей волной.",
            "",
        ]
    ).strip() + "\n"


def build_model_validation_markdown(
    layout: ModelValidationLayoutResult,
    *,
    repeated_splits_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    generalization_summary_df: pd.DataFrame,
    gap_diagnostics_df: pd.DataFrame,
    risk_audit_df: pd.DataFrame,
) -> str:
    """Собрать markdown отчёт heavy validation после repeated splits."""
    request = layout.request
    protocol = layout.protocol
    artifact_paths = layout.artifact_paths
    repeated_states = ", ".join(str(state) for state in protocol.repeated_split.random_states)

    note_block = ""
    if request.note:
        note_block = f"\n## Note\n\n{request.note}\n"

    return "\n".join(
        [
            f"# Model Validation Run: `{request.run_name}`",
            "",
            "## Status",
            "",
            "- stage: `repeated_split_completed`",
            f"- mode: `{request.mode}`",
            f"- include_optional_diagnostics: `{request.include_optional_diagnostics}`",
            f"- split_count: `{len(protocol.repeated_split.random_states)}`",
            "",
            "## Protocol",
            "",
            f"- validation protocol: `{protocol.name}`",
            f"- benchmark protocol: `{protocol.comparison_protocol.name}`",
            f"- repeated split random states: `{repeated_states}`",
            f"- primary metrics: `{', '.join(protocol.primary_metrics)}`",
            note_block.rstrip(),
            "",
            "## Model Summary",
            "",
            frame_to_text(model_summary_df),
            "",
            "## Stage Summary",
            "",
            frame_to_text(generalization_summary_df),
            "",
            "## Gap Diagnostics",
            "",
            frame_to_text(gap_diagnostics_df),
            "",
            "## Per-model Risk Audit",
            "",
            frame_to_text(risk_audit_df),
            "",
            "## Repeated Split Diagnostics",
            "",
            frame_to_text(repeated_splits_df),
            "",
            "## Artifacts",
            "",
            f"- report: `{artifact_paths.report_markdown_path.name}`",
            f"- repeated splits: `{artifact_paths.repeated_splits_csv_path.name}`",
            f"- model summary: `{artifact_paths.model_summary_csv_path.name}`",
            f"- generalization summary: `{artifact_paths.generalization_summary_csv_path.name}`",
            f"- gap diagnostics: `{artifact_paths.gap_diagnostics_csv_path.name}`",
            f"- risk audit: `{artifact_paths.risk_audit_csv_path.name}`",
            f"- plots dir: `{artifact_paths.plots_dir.name}`",
            "",
        ]
    ).strip() + "\n"


def save_model_validation_artifacts(
    layout: ModelValidationLayoutResult,
    *,
    repeated_splits_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    generalization_summary_df: pd.DataFrame,
    gap_diagnostics_df: pd.DataFrame,
    risk_audit_df: pd.DataFrame,
) -> None:
    """Сохранить markdown и CSV артефакты heavy validation run."""
    artifact_paths = layout.artifact_paths
    repeated_splits_df.to_csv(artifact_paths.repeated_splits_csv_path, index=False)
    model_summary_df.to_csv(artifact_paths.model_summary_csv_path, index=False)
    generalization_summary_df.to_csv(
        artifact_paths.generalization_summary_csv_path,
        index=False,
    )
    gap_diagnostics_df.to_csv(
        artifact_paths.gap_diagnostics_csv_path,
        index=False,
    )
    risk_audit_df.to_csv(
        artifact_paths.risk_audit_csv_path,
        index=False,
    )
    artifact_paths.report_markdown_path.write_text(
        build_model_validation_markdown(
            layout,
            repeated_splits_df=repeated_splits_df,
            model_summary_df=model_summary_df,
            generalization_summary_df=generalization_summary_df,
            gap_diagnostics_df=gap_diagnostics_df,
            risk_audit_df=risk_audit_df,
        ),
        encoding="utf-8",
    )


__all__ = [
    "build_model_validation_markdown",
    "build_validation_layout_markdown",
    "frame_to_text",
    "save_model_validation_artifacts",
]
