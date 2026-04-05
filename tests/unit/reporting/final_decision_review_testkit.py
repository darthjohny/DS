# Тестовый файл `final_decision_review_testkit.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from numbers import Integral

import pandas as pd


def require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def build_decision_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "hostname": ["Star A", pd.NA, pd.NA],
            "ra": [10.0, 20.0, 30.0],
            "dec": [-1.0, 2.0, 4.0],
            "quality_reason": ["clean", "high_ruwe", "missing_core_features"],
            "review_bucket": ["pass", "review_high_ruwe", "reject_missing_core_features"],
            "host_similarity_score": [0.92, 0.35, 0.15],
            "teff_gspphot": [5800.0, 4700.0, 3900.0],
            "logg_gspphot": [4.4, 3.1, 4.7],
            "mh_gspphot": [0.1, -0.3, -0.1],
            "parallax": [10.0, 6.0, 3.0],
            "parallax_over_error": [12.0, 4.0, 2.0],
            "ruwe": [1.0, 1.3, 1.8],
            "phot_g_mean_mag": [11.0, 13.0, 15.0],
            "radius_flame": [1.0, 4.0, pd.NA],
            "radius_gspphot": [1.1, 3.9, pd.NA],
            "lum_flame": [1.0, 20.0, pd.NA],
            "evolstage_flame": ["MainSequence", "RGB", pd.NA],
        }
    )


def build_final_decision_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "final_domain_state": ["id", "candidate_ood", "unknown"],
            "final_quality_state": ["pass", "pass", "reject"],
            "final_coarse_class": ["G", "K", pd.NA],
            "final_coarse_confidence": [0.98, 0.61, pd.NA],
            "final_refinement_label": ["G2", pd.NA, pd.NA],
            "final_refinement_state": ["accepted", "not_attempted", "rejected_to_unknown"],
            "final_refinement_confidence": [0.85, pd.NA, pd.NA],
            "final_decision_reason": [
                "refinement_accepted",
                "candidate_ood_kept",
                "quality_reject",
            ],
            "priority_state": ["high", pd.NA, pd.NA],
        }
    )


def build_priority_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2],
            "spec_class": ["G", "K"],
            "host_similarity_score": [0.92, 0.12],
        }
    )


def build_priority_ranking_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2],
            "priority_score": [0.83, 0.28],
            "priority_label": ["high", "low"],
            "priority_reason": ["strong host-like signal", "weak host-like signal"],
            "class_priority_score": [1.0, 0.2],
            "host_similarity_score": [0.92, 0.12],
            "observability_score": [0.75, 0.41],
            "brightness_score": [0.81, 0.39],
            "distance_score": [0.73, 0.18],
            "astrometry_score": [0.71, 0.64],
        }
    )
