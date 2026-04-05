# Файл `priority_score_contracts.py` слоя `ranking`.
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
from typing import Literal

type PriorityLabel = Literal["low", "medium", "high"]

DEFAULT_HOST_SCORE_COLUMN = "host_similarity_score"
RANKING_REQUIRED_COLUMNS: tuple[str, ...] = ("source_id", "spec_class")
RANKING_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "evolution_stage",
    "parallax",
    "phot_g_mean_mag",
    "parallax_over_error",
    "ruwe",
    "validation_factor",
)


@dataclass(frozen=True, slots=True)
class RankingWeights:
    # Веса итогового priority-score.
    class_priority_weight: float = 0.20
    host_similarity_weight: float = 0.45
    observability_weight: float = 0.35

    def __post_init__(self) -> None:
        # Для explainable scoring веса должны быть положительными и суммироваться в 1.
        total_weight = (
            self.class_priority_weight
            + self.host_similarity_weight
            + self.observability_weight
        )
        if min(
            self.class_priority_weight,
            self.host_similarity_weight,
            self.observability_weight,
        ) <= 0.0:
            raise ValueError("Ranking weights must be positive.")
        if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Ranking weights must sum to 1.0.")


@dataclass(frozen=True, slots=True)
class PriorityThresholds:
    # Пороговые правила первой волны.
    high_min: float = 0.75
    medium_min: float = 0.45
    low_priority_class_cap: float = 0.34
    evolved_stage_penalty: float = 0.15

    def __post_init__(self) -> None:
        # Пороговая схема должна оставаться монотонной и читаемой.
        if not 0.0 <= self.low_priority_class_cap <= 1.0:
            raise ValueError("low_priority_class_cap must be between 0 and 1.")
        if not 0.0 <= self.medium_min <= self.high_min <= 1.0:
            raise ValueError("Priority thresholds must satisfy 0 <= medium <= high <= 1.")
        if not 0.0 <= self.evolved_stage_penalty <= 1.0:
            raise ValueError("evolved_stage_penalty must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class PriorityScoreRecord:
    # Развернутая запись итогового приоритета по объекту.
    source_id: str
    spec_class: str
    evolution_stage: str | None
    class_priority_score: float
    host_similarity_score: float
    brightness_score: float
    distance_score: float
    astrometry_score: float
    observability_score: float
    brightness_available: bool
    distance_available: bool
    astrometry_available: bool
    observability_evidence_count: int
    priority_score: float
    priority_label: PriorityLabel
    priority_reason: str


DEFAULT_RANKING_WEIGHTS = RankingWeights()
DEFAULT_PRIORITY_THRESHOLDS = PriorityThresholds()


__all__ = [
    "DEFAULT_HOST_SCORE_COLUMN",
    "DEFAULT_PRIORITY_THRESHOLDS",
    "DEFAULT_RANKING_WEIGHTS",
    "PriorityLabel",
    "PriorityScoreRecord",
    "PriorityThresholds",
    "RANKING_OPTIONAL_COLUMNS",
    "RANKING_REQUIRED_COLUMNS",
    "RankingWeights",
]
