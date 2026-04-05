# Файл `coarse_ob_domain_shift_review.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from exohost.reporting.coarse_ob_domain_shift_bundle import (
    build_coarse_ob_domain_shift_review_bundle,
    build_downstream_ob_boundary_frame,
    build_train_time_ob_boundary_frame,
    load_coarse_ob_domain_shift_review_bundle_from_env,
)
from exohost.reporting.coarse_ob_domain_shift_contracts import (
    CoarseOBDomainShiftConfig,
    CoarseOBDomainShiftReviewBundle,
)
from exohost.reporting.coarse_ob_domain_shift_frames import (
    build_domain_class_balance_frame,
    build_domain_confusion_frame,
    build_domain_membership_summary_frame,
    build_domain_missingness_summary_frame,
    build_domain_physics_summary_frame,
    build_domain_predicted_class_summary_frame,
    build_domain_probability_summary_frame,
    build_domain_shift_auc_frame,
)

__all__ = [
    "CoarseOBDomainShiftConfig",
    "CoarseOBDomainShiftReviewBundle",
    "build_coarse_ob_domain_shift_review_bundle",
    "build_domain_class_balance_frame",
    "build_domain_confusion_frame",
    "build_domain_membership_summary_frame",
    "build_domain_missingness_summary_frame",
    "build_domain_physics_summary_frame",
    "build_domain_predicted_class_summary_frame",
    "build_domain_probability_summary_frame",
    "build_domain_shift_auc_frame",
    "build_downstream_ob_boundary_frame",
    "build_train_time_ob_boundary_frame",
    "load_coarse_ob_domain_shift_review_bundle_from_env",
]
