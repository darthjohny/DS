# Файл `priority_threshold_review_contracts.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from dataclasses import dataclass

from exohost.ranking.priority_score import (
    DEFAULT_PRIORITY_THRESHOLDS,
    PriorityThresholds,
)


@dataclass(frozen=True, slots=True)
class PriorityThresholdVariant:
    # Один именованный threshold-variant для review notebook.
    name: str
    thresholds: PriorityThresholds


DEFAULT_PRIORITY_THRESHOLD_VARIANTS: tuple[PriorityThresholdVariant, ...] = (
    PriorityThresholdVariant(
        name="baseline",
        thresholds=DEFAULT_PRIORITY_THRESHOLDS,
    ),
    PriorityThresholdVariant(
        name="strict_high_080",
        thresholds=PriorityThresholds(
            high_min=0.80,
            medium_min=0.45,
            low_priority_class_cap=DEFAULT_PRIORITY_THRESHOLDS.low_priority_class_cap,
            evolved_stage_penalty=DEFAULT_PRIORITY_THRESHOLDS.evolved_stage_penalty,
        ),
    ),
    PriorityThresholdVariant(
        name="strict_high_medium_085_055",
        thresholds=PriorityThresholds(
            high_min=0.85,
            medium_min=0.55,
            low_priority_class_cap=DEFAULT_PRIORITY_THRESHOLDS.low_priority_class_cap,
            evolved_stage_penalty=DEFAULT_PRIORITY_THRESHOLDS.evolved_stage_penalty,
        ),
    ),
)


__all__ = [
    "DEFAULT_PRIORITY_THRESHOLD_VARIANTS",
    "PriorityThresholdVariant",
]
