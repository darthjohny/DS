# Файл `scoring_review_contracts.py` слоя `reporting`.
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
from typing import Any

import pandas as pd

TOP_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "source_id",
    "spec_class",
    "evolution_stage",
    "priority_score",
    "priority_label",
    "host_similarity_score",
    "observability_score",
    "observability_evidence_count",
    "priority_reason",
)


@dataclass(frozen=True, slots=True)
class ScoringReviewBundle:
    # Полный пакет scoring/ranking артефактов для notebook-review слоя.
    scoring_df: pd.DataFrame
    scoring_metadata: dict[str, Any]
    ranking_df: pd.DataFrame | None = None
    ranking_metadata: dict[str, Any] | None = None


__all__ = [
    "ScoringReviewBundle",
    "TOP_CANDIDATE_COLUMNS",
]
