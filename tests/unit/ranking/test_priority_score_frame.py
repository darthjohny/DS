# Тестовый файл `test_priority_score_frame.py` домена `ranking`.
#
# Этот файл проверяет только:
# - проверку логики домена: priority- и observability-логики;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `ranking` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd
import pytest

from exohost.ranking.priority_score import (
    build_priority_ranking_frame,
)

from .priority_score_testkit import build_candidate_frame, get_float_cell, get_str_cell


def test_build_priority_ranking_frame_sorts_best_candidate_first() -> None:
    # Сильный G-dwarf с хорошей наблюдаемостью должен оказаться наверху.
    ranking_frame = build_priority_ranking_frame(build_candidate_frame())
    best_priority_score = get_float_cell(ranking_frame, 0, "priority_score")
    next_priority_score = get_float_cell(ranking_frame, 1, "priority_score")

    assert get_str_cell(ranking_frame, 0, "source_id") == "1"
    assert get_str_cell(ranking_frame, 0, "priority_label") == "high"
    assert best_priority_score > next_priority_score


def test_build_priority_ranking_frame_rejects_missing_host_score_column() -> None:
    # Ranking-слой не должен молча продолжать работу без host-like сигнала.
    candidate_frame = build_candidate_frame().drop(columns="host_similarity_score")

    with pytest.raises(ValueError, match="missing required columns"):
        build_priority_ranking_frame(candidate_frame)


def test_build_priority_ranking_frame_uses_neutral_fallback_for_missing_observability() -> None:
    # Если observability-поля отсутствуют, scoring должен оставаться устойчивым.
    candidate_frame = pd.DataFrame(
        [
            {
                "source_id": "42",
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "host_similarity_score": 0.80,
            }
        ]
    )

    ranking_frame = build_priority_ranking_frame(candidate_frame)
    observability_score = get_float_cell(ranking_frame, 0, "observability_score")

    assert observability_score == 0.50
    assert get_float_cell(ranking_frame, 0, "observability_evidence_count") == 0.0
    assert get_str_cell(ranking_frame, 0, "priority_label") == "medium"
    assert "нейтральному fallback" in get_str_cell(ranking_frame, 0, "priority_reason")
