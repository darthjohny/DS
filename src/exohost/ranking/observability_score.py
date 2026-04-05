# Файл `observability_score.py` слоя `ranking`.
#
# Этот файл отвечает только за:
# - логики приоритизации и наблюдательной пригодности;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ranking` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import math
from dataclasses import dataclass

NEUTRAL_SCORE = 0.50


@dataclass(frozen=True, slots=True)
class ObservabilityScoreRecord:
    # Компоненты наблюдательной пригодности объекта.
    brightness_score: float
    distance_score: float
    astrometry_score: float
    observability_score: float
    brightness_available: bool
    distance_available: bool
    astrometry_available: bool
    observability_evidence_count: int


def clamp_score(value: float) -> float:
    # Жестко ограничиваем score в диапазоне [0, 1].
    return max(0.0, min(1.0, float(value)))


def is_available_signal(value: float | None) -> bool:
    # Проверяем, есть ли у нас реальный числовой сигнал, а не пропуск.
    return value is not None and not math.isnan(value)


def build_linear_score(
    value: float | None,
    *,
    low: float,
    high: float,
    higher_is_better: bool,
    missing_value: float = NEUTRAL_SCORE,
) -> float:
    # Переводим физический признак в простую линейную шкалу [0, 1].
    if high <= low:
        raise ValueError("Linear score range must satisfy high > low.")

    if value is None or math.isnan(value):
        return missing_value

    normalized_value = (float(value) - low) / (high - low)
    score = normalized_value if higher_is_better else 1.0 - normalized_value
    return clamp_score(score)


def compute_brightness_score(phot_g_mean_mag: float | None) -> float:
    # Чем ярче объект в G-band, тем проще последующее наблюдение.
    return build_linear_score(
        phot_g_mean_mag,
        low=10.0,
        high=16.0,
        higher_is_better=False,
    )


def compute_distance_score(parallax: float | None) -> float:
    # На первой волне используем параллакс как грубый proxy близости объекта.
    return build_linear_score(
        parallax,
        low=2.0,
        high=20.0,
        higher_is_better=True,
    )


def compute_astrometry_score(
    parallax_over_error: float | None,
    ruwe: float | None,
    validation_factor: float | None,
) -> float:
    # Собираем quality score только из реально доступных сигналов надежности.
    component_scores: list[float] = []

    if parallax_over_error is not None and not math.isnan(parallax_over_error):
        component_scores.append(
            build_linear_score(
                parallax_over_error,
                low=5.0,
                high=20.0,
                higher_is_better=True,
            )
        )

    if ruwe is not None and not math.isnan(ruwe):
        component_scores.append(
            build_linear_score(
                ruwe,
                low=1.0,
                high=1.4,
                higher_is_better=False,
            )
        )

    if validation_factor is not None and not math.isnan(validation_factor):
        component_scores.append(clamp_score(validation_factor))

    if not component_scores:
        return NEUTRAL_SCORE

    return clamp_score(sum(component_scores) / len(component_scores))


def build_observability_score_record(
    *,
    phot_g_mean_mag: float | None,
    parallax: float | None,
    parallax_over_error: float | None,
    ruwe: float | None,
    validation_factor: float | None,
) -> ObservabilityScoreRecord:
    # Объединяем яркость, близость и качество астрометрии в единый observability score.
    brightness_available = is_available_signal(phot_g_mean_mag)
    distance_available = is_available_signal(parallax)
    astrometry_available = any(
        (
            is_available_signal(parallax_over_error),
            is_available_signal(ruwe),
            is_available_signal(validation_factor),
        )
    )
    brightness_score = compute_brightness_score(phot_g_mean_mag)
    distance_score = compute_distance_score(parallax)
    astrometry_score = compute_astrometry_score(
        parallax_over_error,
        ruwe,
        validation_factor,
    )
    observability_evidence_count = int(brightness_available) + int(distance_available) + int(astrometry_available)
    observability_score = clamp_score(
        0.40 * brightness_score
        + 0.35 * distance_score
        + 0.25 * astrometry_score
    )
    return ObservabilityScoreRecord(
        brightness_score=brightness_score,
        distance_score=distance_score,
        astrometry_score=astrometry_score,
        observability_score=observability_score,
        brightness_available=brightness_available,
        distance_available=distance_available,
        astrometry_available=astrometry_available,
        observability_evidence_count=observability_evidence_count,
    )
