# Файл `ranking_report.py` слоя `reporting`.
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
from typing import Any

import pandas as pd

from exohost.reporting.markdown_tables import format_scalar_line, frame_to_code_block
from exohost.reporting.ranking_artifacts import load_ranking_artifacts

RANKING_REPORT_FILENAME = "summary.md"
RANKING_PREVIEW_COLUMNS: tuple[str, ...] = (
    "source_id",
    "spec_class",
    "evolution_stage",
    "priority_score",
    "priority_label",
    "host_similarity_score",
    "observability_score",
    "priority_reason",
)

def build_ranking_summary_lines(
    ranking_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[str]:
    # Собираем краткое summary ranking-прогона.
    summary_lines = [
        format_scalar_line("ranking_name", metadata.get("ranking_name", "unknown")),
        format_scalar_line("created_at_utc", metadata.get("created_at_utc", "unknown")),
        format_scalar_line("n_rows", metadata.get("n_rows", int(ranking_df.shape[0]))),
    ]

    label_distribution = metadata.get("priority_label_distribution")
    if isinstance(label_distribution, dict) and label_distribution:
        formatted_distribution = ", ".join(
            f"{label}={count}"
            for label, count in sorted(label_distribution.items())
        )
        summary_lines.append(format_scalar_line("priority_label_distribution", formatted_distribution))

    if not ranking_df.empty and "source_id" in ranking_df.columns:
        top_source_id = str(ranking_df.iloc[0]["source_id"])
        summary_lines.append(format_scalar_line("top_source_id", top_source_id))

    return summary_lines


def build_ranking_report_markdown(
    ranking_df: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    top_rows: int = 10,
) -> str:
    # Собираем markdown-отчет по ranking-прогону.
    preview_columns = [name for name in RANKING_PREVIEW_COLUMNS if name in ranking_df.columns]
    preview_df = ranking_df.loc[:, preview_columns].head(top_rows).copy()

    sections = [
        "# Ranking Report",
        "",
        *build_ranking_summary_lines(ranking_df, metadata),
        "",
        "## Top Candidates",
        frame_to_code_block(preview_df),
    ]
    return "\n".join(sections).strip() + "\n"


def save_ranking_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    top_rows: int = 10,
) -> Path:
    # Сохраняем markdown-отчет рядом с ranking-артефактами.
    ranking_dir = Path(run_dir)
    ranking_df, metadata = load_ranking_artifacts(ranking_dir)
    report_text = build_ranking_report_markdown(
        ranking_df,
        metadata,
        top_rows=top_rows,
    )
    report_path = Path(output_path) if output_path is not None else ranking_dir / RANKING_REPORT_FILENAME
    report_path.write_text(report_text, encoding="utf-8")
    return report_path
