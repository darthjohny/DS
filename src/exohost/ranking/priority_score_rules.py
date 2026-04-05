# Файл `priority_score_rules.py` слоя `ranking`.
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

import pandas as pd

from exohost.contracts.label_contract import (
    LOW_PRIORITY_SPECTRAL_CLASSES,
    TARGET_SPECTRAL_CLASSES,
    normalize_evolution_stage,
)
from exohost.ranking.observability_score import (
    NEUTRAL_SCORE,
    ObservabilityScoreRecord,
    clamp_score,
)
from exohost.ranking.priority_score_contracts import (
    DEFAULT_PRIORITY_THRESHOLDS,
    RANKING_REQUIRED_COLUMNS,
    PriorityLabel,
    PriorityThresholds,
)


def require_ranking_columns(
    df: pd.DataFrame,
    *,
    host_score_column: str,
) -> None:
    # Проверяем минимальный контракт ranking-слоя.
    required_columns = (*RANKING_REQUIRED_COLUMNS, host_score_column)
    missing_columns = [name for name in required_columns if name not in df.columns]
    if missing_columns:
        missing_sql = ", ".join(missing_columns)
        raise ValueError(f"Ranking frame is missing required columns: {missing_sql}")


def compute_class_priority_score(
    spec_class: str,
    evolution_stage: str | None,
    *,
    thresholds: PriorityThresholds = DEFAULT_PRIORITY_THRESHOLDS,
) -> float:
    # На первой волне coarse-class уже сам задает сильный prior.
    normalized_class = str(spec_class).strip().upper()
    normalized_stage = (
        normalize_evolution_stage(evolution_stage)
        if evolution_stage is not None
        else None
    )

    if normalized_class in TARGET_SPECTRAL_CLASSES:
        class_priority_score = 1.0
    elif normalized_class in LOW_PRIORITY_SPECTRAL_CLASSES:
        class_priority_score = 0.20
    else:
        class_priority_score = NEUTRAL_SCORE

    if normalized_class in TARGET_SPECTRAL_CLASSES and normalized_stage == "evolved":
        class_priority_score -= thresholds.evolved_stage_penalty

    return clamp_score(class_priority_score)


def compute_host_similarity_score(value: float | None) -> float:
    # Host score на входе уже должен быть вероятностью или близкой шкалой.
    if value is None or math.isnan(value):
        return NEUTRAL_SCORE
    return clamp_score(value)


def assign_priority_label(
    priority_score: float,
    *,
    thresholds: PriorityThresholds = DEFAULT_PRIORITY_THRESHOLDS,
) -> PriorityLabel:
    # Переводим численный скоринг в человекочитаемую категорию.
    if priority_score >= thresholds.high_min:
        return "high"
    if priority_score >= thresholds.medium_min:
        return "medium"
    return "low"


def build_priority_reason(
    *,
    spec_class: str,
    evolution_stage: str | None,
    host_similarity_score: float,
    observability_record: ObservabilityScoreRecord,
) -> str:
    # Собираем короткое текстовое объяснение приоритета.
    normalized_class = str(spec_class).strip().upper()
    normalized_stage = (
        normalize_evolution_stage(evolution_stage)
        if evolution_stage is not None
        else None
    )
    reason_parts: list[str] = []

    if normalized_class in LOW_PRIORITY_SPECTRAL_CLASSES:
        reason_parts.append("упрощенная low-priority ветка по спектральному классу")
    elif normalized_stage == "evolved":
        reason_parts.append("штраф за evolved-стадию")

    if host_similarity_score >= 0.75:
        reason_parts.append("сильный host-like сигнал")
    elif host_similarity_score <= 0.35:
        reason_parts.append("слабый host-like сигнал")

    if observability_record.observability_score >= 0.70:
        reason_parts.append("хорошая наблюдательная пригодность")
    elif observability_record.observability_score <= 0.35:
        reason_parts.append("ограниченная наблюдательная пригодность")

    if observability_record.observability_evidence_count == 0:
        reason_parts.append(
            "observability рассчитана по нейтральному fallback без входных сигналов"
        )
    elif observability_record.observability_evidence_count < 3:
        reason_parts.append("observability рассчитана по неполному набору сигналов")

    if not reason_parts:
        reason_parts.append("сбалансированный профиль без явных экстремумов")

    return "; ".join(reason_parts)


__all__ = [
    "assign_priority_label",
    "build_priority_reason",
    "compute_class_priority_score",
    "compute_host_similarity_score",
    "require_ranking_columns",
]
