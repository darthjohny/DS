# Тестовый файл `test_priority_score_rules.py` домена `ranking`.
#
# Этот файл проверяет только:
# - проверку логики домена: priority- и observability-логики;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `ranking` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.ranking.priority_score import (
    build_priority_score_record,
    compute_host_similarity_score,
)

from .priority_score_testkit import build_candidate_frame


def test_build_priority_score_record_caps_low_priority_spectral_classes() -> None:
    # O/B/A на первой волне не должны подниматься выше low-priority ветки.
    row = build_candidate_frame().iloc[2]

    record = build_priority_score_record(row)

    assert record.spec_class == "A"
    assert record.priority_label == "low"
    assert record.priority_score <= 0.34
    assert "low-priority" in record.priority_reason


def test_compute_host_similarity_score_returns_neutral_for_missing_input() -> None:
    assert compute_host_similarity_score(None) == 0.50
