# Файл `priority_score.py` слоя `ranking`.
#
# Этот файл отвечает только за:
# - логики приоритизации и наблюдательной пригодности;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ranking` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.ranking.priority_score_contracts import (
    DEFAULT_HOST_SCORE_COLUMN,
    DEFAULT_PRIORITY_THRESHOLDS,
    DEFAULT_RANKING_WEIGHTS,
    RANKING_OPTIONAL_COLUMNS,
    RANKING_REQUIRED_COLUMNS,
    PriorityLabel,
    PriorityScoreRecord,
    PriorityThresholds,
    RankingWeights,
)
from exohost.ranking.priority_score_frame import (
    build_priority_ranking_frame,
    build_priority_score_record,
    priority_record_to_dict,
)
from exohost.ranking.priority_score_rules import (
    assign_priority_label,
    build_priority_reason,
    compute_class_priority_score,
    compute_host_similarity_score,
    require_ranking_columns,
)
from exohost.ranking.priority_score_scalars import (
    coerce_optional_float,
    is_missing_scalar,
)

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
    "assign_priority_label",
    "build_priority_ranking_frame",
    "build_priority_reason",
    "build_priority_score_record",
    "coerce_optional_float",
    "compute_class_priority_score",
    "compute_host_similarity_score",
    "is_missing_scalar",
    "priority_record_to_dict",
    "require_ranking_columns",
]
