# Тестовый файл `test_ranking_report.py` домена `reporting`.
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

from exohost.reporting.ranking_artifacts import save_ranking_artifacts
from exohost.reporting.ranking_report import save_ranking_report


def build_ranking_frame() -> pd.DataFrame:
    # Небольшая synthetic ranking-таблица для report-слоя.
    return pd.DataFrame(
        [
            {
                "source_id": "1",
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "priority_score": 0.92,
                "priority_label": "high",
                "host_similarity_score": 0.90,
                "observability_score": 0.88,
                "priority_reason": "сильный host-like сигнал",
            },
            {
                "source_id": "2",
                "spec_class": "K",
                "evolution_stage": "evolved",
                "priority_score": 0.58,
                "priority_label": "medium",
                "host_similarity_score": 0.61,
                "observability_score": 0.55,
                "priority_reason": "штраф за evolved-стадию",
            },
        ]
    )


def test_save_ranking_report_creates_summary_markdown(tmp_path: Path) -> None:
    # Проверяем сохранение markdown-summary рядом с ranking-артефактами.
    run_paths = save_ranking_artifacts(
        build_ranking_frame(),
        ranking_name="router_candidates",
        output_dir=tmp_path,
        now=datetime(2026, 3, 20, 10, 5, 0, tzinfo=UTC),
    )

    report_path = save_ranking_report(run_paths.run_dir, top_rows=5)
    report_text = report_path.read_text(encoding="utf-8")

    assert report_path.name == "summary.md"
    assert "# Ranking Report" in report_text
    assert "priority_label_distribution" in report_text
    assert "Top Candidates" in report_text
