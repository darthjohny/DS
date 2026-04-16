# Файл `scoring_review.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from exohost.reporting.scoring_review_bundle import load_scoring_review_bundle
from exohost.reporting.scoring_review_contracts import ScoringReviewBundle
from exohost.reporting.scoring_review_frames import (
    build_goal_alignment_frame,
    build_observability_coverage_frame,
    build_prediction_distribution_frame,
    build_priority_distribution_frame,
    build_scoring_summary_frame,
    build_top_candidates_frame,
)

# Тонкий публичный фасад для scoring-review слоя.
# Через него notebook и docs получают все основные сводки без знания внутренней
# структуры bundle/contracts/frames модулей.
__all__ = [
    "ScoringReviewBundle",
    "build_goal_alignment_frame",
    "build_observability_coverage_frame",
    "build_prediction_distribution_frame",
    "build_priority_distribution_frame",
    "build_scoring_summary_frame",
    "build_top_candidates_frame",
    "load_scoring_review_bundle",
]
