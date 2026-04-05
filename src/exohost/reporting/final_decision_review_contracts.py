# Файл `final_decision_review_contracts.py` слоя `reporting`.
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
from pathlib import Path
from typing import Any

import pandas as pd

TOP_PRIORITY_COLUMNS: tuple[str, ...] = (
    "source_id",
    "final_domain_state",
    "final_quality_state",
    "final_coarse_class",
    "final_refinement_label",
    "priority_score",
    "priority_label",
    "priority_reason",
)

HIGH_PRIORITY_PHYSICS_COLUMNS: tuple[str, ...] = (
    "source_id",
    "final_coarse_class",
    "final_refinement_label",
    "priority_score",
    "priority_reason",
    "host_similarity_score",
    "observability_score",
    "class_priority_score",
    "brightness_score",
    "distance_score",
    "astrometry_score",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "phot_g_mean_mag",
    "radius_flame",
    "lum_flame",
)

STAR_RESULT_COLUMNS: tuple[str, ...] = (
    "source_id",
    "hostname",
    "ra",
    "dec",
    "final_domain_state",
    "final_quality_state",
    "final_coarse_class",
    "final_coarse_confidence",
    "final_refinement_label",
    "final_refinement_state",
    "final_refinement_confidence",
    "final_decision_reason",
    "quality_reason",
    "review_bucket",
    "priority_score",
    "priority_label",
    "priority_reason",
    "host_similarity_score",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "phot_g_mean_mag",
    "radius_flame",
    "radius_gspphot",
    "lum_flame",
    "evolstage_flame",
)


@dataclass(frozen=True, slots=True)
class FinalDecisionReviewBundle:
    # Полный пакет final decision artifacts для notebook-review слоя.
    run_dir: Path
    decision_input_df: pd.DataFrame
    final_decision_df: pd.DataFrame
    priority_input_df: pd.DataFrame
    priority_ranking_df: pd.DataFrame
    metadata: dict[str, Any]


__all__ = [
    "FinalDecisionReviewBundle",
    "HIGH_PRIORITY_PHYSICS_COLUMNS",
    "STAR_RESULT_COLUMNS",
    "TOP_PRIORITY_COLUMNS",
]
