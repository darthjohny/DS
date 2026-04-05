# Тестовый файл `test_observability_score.py` домена `ranking`.
#
# Этот файл проверяет только:
# - проверку логики домена: priority- и observability-логики;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `ranking` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.ranking.observability_score import (
    NEUTRAL_SCORE,
    build_observability_score_record,
)


def test_build_observability_score_record_returns_neutral_value_for_missing_inputs() -> None:
    # При отсутствии признаков оставляем нейтральный observability score.
    record = build_observability_score_record(
        phot_g_mean_mag=None,
        parallax=None,
        parallax_over_error=None,
        ruwe=None,
        validation_factor=None,
    )

    assert record.brightness_score == NEUTRAL_SCORE
    assert record.distance_score == NEUTRAL_SCORE
    assert record.astrometry_score == NEUTRAL_SCORE
    assert record.observability_score == NEUTRAL_SCORE
    assert record.brightness_available is False
    assert record.distance_available is False
    assert record.astrometry_available is False
    assert record.observability_evidence_count == 0


def test_build_observability_score_record_prefers_bright_close_high_quality_targets() -> None:
    # Яркий, близкий и качественный объект должен получать высокий score.
    record = build_observability_score_record(
        phot_g_mean_mag=10.3,
        parallax=18.0,
        parallax_over_error=22.0,
        ruwe=1.01,
        validation_factor=0.95,
    )

    assert record.brightness_score > 0.9
    assert record.distance_score > 0.8
    assert record.astrometry_score > 0.9
    assert record.observability_score > 0.85
    assert record.brightness_available is True
    assert record.distance_available is True
    assert record.astrometry_available is True
    assert record.observability_evidence_count == 3
