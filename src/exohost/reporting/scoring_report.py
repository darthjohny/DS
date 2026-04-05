# Файл `scoring_report.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from pathlib import Path

from exohost.reporting.markdown_tables import format_scalar_line, frame_to_code_block
from exohost.reporting.scoring_review import (
    build_goal_alignment_frame,
    build_observability_coverage_frame,
    build_prediction_distribution_frame,
    build_priority_distribution_frame,
    build_scoring_summary_frame,
    build_top_candidates_frame,
    load_scoring_review_bundle,
)

SCORING_REPORT_FILENAME = "summary.md"


def build_scoring_summary_lines(
    scoring_summary_df,
) -> list[str]:
    # Собираем краткое summary scoring-прогона.
    if scoring_summary_df.empty:
        return []

    row = scoring_summary_df.iloc[0]
    return [
        format_scalar_line("task_name", row["task_name"]),
        format_scalar_line("target_column", row["target_column"]),
        format_scalar_line("model_name", row["model_name"]),
        format_scalar_line("created_at_utc", row["created_at_utc"]),
        format_scalar_line("n_rows", row["n_rows"]),
        format_scalar_line("score_mode", row["score_mode"]),
        format_scalar_line("has_ranking", row["has_ranking"]),
    ]


def build_scoring_report_markdown(
    *,
    scoring_run_dir: str | Path,
    ranking_run_dir: str | Path | None = None,
    top_rows: int = 10,
) -> str:
    # Собираем markdown-отчет по scoring-прогону и связанному ranking-прогону.
    bundle = load_scoring_review_bundle(
        str(scoring_run_dir),
        ranking_run_dir=None if ranking_run_dir is None else str(ranking_run_dir),
    )
    scoring_summary_df = build_scoring_summary_frame(bundle)
    prediction_distribution_df = build_prediction_distribution_frame(bundle)

    sections = [
        "# Scoring Report",
        "",
        *build_scoring_summary_lines(scoring_summary_df),
        "",
        "## Prediction Distribution",
        frame_to_code_block(prediction_distribution_df),
    ]

    if bundle.ranking_df is not None:
        priority_distribution_df = build_priority_distribution_frame(bundle)
        observability_coverage_df = build_observability_coverage_frame(bundle)
        goal_alignment_df = build_goal_alignment_frame(bundle, top_n=top_rows)
        top_candidates_df = build_top_candidates_frame(bundle, top_n=top_rows)
        sections.extend(
            [
                "",
                "## Priority Distribution",
                frame_to_code_block(priority_distribution_df),
                "",
                "## Observability Coverage",
                frame_to_code_block(observability_coverage_df),
                "",
                "## Goal Alignment",
                frame_to_code_block(goal_alignment_df),
                "",
                "## Top Candidates",
                frame_to_code_block(top_candidates_df),
            ]
        )

    return "\n".join(sections).strip() + "\n"


def save_scoring_report(
    scoring_run_dir: str | Path,
    *,
    ranking_run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    top_rows: int = 10,
) -> Path:
    # Сохраняем markdown-отчет рядом со scoring-артефактами.
    scoring_dir = Path(scoring_run_dir)
    report_text = build_scoring_report_markdown(
        scoring_run_dir=scoring_dir,
        ranking_run_dir=ranking_run_dir,
        top_rows=top_rows,
    )
    report_path = Path(output_path) if output_path is not None else scoring_dir / SCORING_REPORT_FILENAME
    report_path.write_text(report_text, encoding="utf-8")
    return report_path
