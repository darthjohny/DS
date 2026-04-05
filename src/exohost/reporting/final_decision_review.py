# Файл `final_decision_review.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from exohost.reporting.final_decision_review_bundle import (
    build_final_decision_summary_frame,
    load_final_decision_review_bundle,
)
from exohost.reporting.final_decision_review_contracts import FinalDecisionReviewBundle
from exohost.reporting.final_decision_review_distributions import (
    build_decision_reason_frame,
    build_domain_distribution_frame,
    build_domain_quality_crosstab_frame,
    build_quality_distribution_frame,
    build_quality_reason_frame,
    build_refinement_distribution_frame,
    build_review_bucket_frame,
)
from exohost.reporting.final_decision_review_priority import (
    build_host_priority_status_frame,
    build_priority_by_coarse_class_frame,
    build_priority_component_quantiles_frame,
    build_priority_distribution_frame,
    build_priority_reason_frame,
    build_top_priority_candidates_frame,
)
from exohost.reporting.final_decision_review_priority_cohort import (
    build_high_priority_candidate_physics_frame,
    build_high_priority_coarse_class_frame,
    build_high_priority_component_summary_frame,
    build_high_priority_refinement_label_frame,
    build_high_priority_summary_frame,
)
from exohost.reporting.final_decision_review_star_level import (
    build_final_coarse_class_frame,
    build_final_refinement_label_frame,
    build_numeric_state_summary_frame,
    build_star_level_result_frame,
    build_star_result_preview_frame,
)

__all__ = [
    "FinalDecisionReviewBundle",
    "build_decision_reason_frame",
    "build_domain_distribution_frame",
    "build_domain_quality_crosstab_frame",
    "build_final_coarse_class_frame",
    "build_final_decision_summary_frame",
    "build_final_refinement_label_frame",
    "build_host_priority_status_frame",
    "build_high_priority_candidate_physics_frame",
    "build_high_priority_coarse_class_frame",
    "build_high_priority_component_summary_frame",
    "build_high_priority_refinement_label_frame",
    "build_high_priority_summary_frame",
    "build_numeric_state_summary_frame",
    "build_priority_by_coarse_class_frame",
    "build_priority_component_quantiles_frame",
    "build_priority_distribution_frame",
    "build_priority_reason_frame",
    "build_quality_distribution_frame",
    "build_quality_reason_frame",
    "build_refinement_distribution_frame",
    "build_review_bucket_frame",
    "build_star_level_result_frame",
    "build_star_result_preview_frame",
    "build_top_priority_candidates_frame",
    "load_final_decision_review_bundle",
]
