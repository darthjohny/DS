# Тестовый файл `test_scoring_report.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from exohost.models.inference import ModelScoringResult
from exohost.reporting.ranking_artifacts import save_ranking_artifacts
from exohost.reporting.scoring_artifacts import save_scoring_artifacts
from exohost.reporting.scoring_report import (
    build_scoring_report_markdown,
    save_scoring_report,
)


def build_scoring_result() -> ModelScoringResult:
    # Небольшой synthetic scoring-результат для markdown-отчета.
    scored_df = pd.DataFrame(
        [
            {
                "source_id": "1",
                "predicted_host_label": "host",
                "predicted_host_label_confidence": 0.91,
                "probability__field": 0.09,
                "probability__host": 0.91,
                "host_similarity_score": 0.91,
                "spec_class": "G",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": "2",
                "predicted_host_label": "field",
                "predicted_host_label_confidence": 0.66,
                "probability__field": 0.66,
                "probability__host": 0.34,
                "host_similarity_score": 0.34,
                "spec_class": "A",
                "evolution_stage": "dwarf",
            },
        ]
    )
    return ModelScoringResult(
        task_name="host_field_classification",
        target_column="host_label",
        model_name="hist_gradient_boosting",
        n_rows=2,
        scored_df=scored_df,
    )


def test_build_scoring_report_markdown_without_ranking(tmp_path: Path) -> None:
    # Проверяем базовый scoring-отчет без ranking-блока.
    scoring_paths = save_scoring_artifacts(
        build_scoring_result(),
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 15, 0, 0, tzinfo=UTC),
    )

    markdown = build_scoring_report_markdown(
        scoring_run_dir=scoring_paths.run_dir,
        top_rows=5,
    )

    assert "# Scoring Report" in markdown
    assert "## Prediction Distribution" in markdown
    assert "## Goal Alignment" not in markdown


def test_build_scoring_report_markdown_with_ranking(tmp_path: Path) -> None:
    # Проверяем scoring-отчет с подключенным ranking-блоком.
    scoring_paths = save_scoring_artifacts(
        build_scoring_result(),
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 15, 1, 0, tzinfo=UTC),
    )
    ranking_paths = save_ranking_artifacts(
        pd.DataFrame(
            [
                {
                    "source_id": "1",
                    "spec_class": "G",
                    "evolution_stage": "dwarf",
                    "priority_score": 0.88,
                    "priority_label": "high",
                    "host_similarity_score": 0.91,
                    "observability_score": 0.85,
                    "priority_reason": "target class with strong scores",
                }
            ]
        ),
        ranking_name="host_candidates",
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 15, 2, 0, tzinfo=UTC),
    )

    markdown = build_scoring_report_markdown(
        scoring_run_dir=scoring_paths.run_dir,
        ranking_run_dir=ranking_paths.run_dir,
        top_rows=5,
    )

    assert "## Priority Distribution" in markdown
    assert "## Observability Coverage" in markdown
    assert "## Goal Alignment" in markdown
    assert "## Top Candidates" in markdown


def test_save_scoring_report_writes_summary(tmp_path: Path) -> None:
    # Проверяем сохранение markdown-отчета рядом со scoring-артефактами.
    scoring_paths = save_scoring_artifacts(
        build_scoring_result(),
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 15, 3, 0, tzinfo=UTC),
    )

    report_path = save_scoring_report(scoring_paths.run_dir)

    assert report_path.exists()
    assert report_path.name == "summary.md"
