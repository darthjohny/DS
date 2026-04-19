# Тестовый helper `ui_testkit.py` домена `ui`.
#
# Этот файл хранит только:
# - компактный in-memory bundle для unit-тестов интерфейсного слоя;
# - повторно используемые dataframe без файлового ввода-вывода.
#
# Следующий слой:
# - отдельные unit-тесты helper-модулей `ui`;
# - сценарные проверки страниц и компонентов.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.reporting.final_decision_artifacts import LoadedFinalDecisionArtifacts
from exohost.ui.loaders import UiLoadedRunBundle


def build_ui_loaded_run_bundle() -> UiLoadedRunBundle:
    # В testkit держим маленький, но реалистичный run bundle с ranked и unknown объектами.
    return UiLoadedRunBundle(
        run_dir=Path("artifacts/decisions/hierarchical_final_decision_demo"),
        loaded_artifacts=LoadedFinalDecisionArtifacts(
            decision_input_df=_build_decision_input_df(),
            final_decision_df=_build_final_decision_df(),
            priority_input_df=_build_priority_input_df(),
            priority_ranking_df=_build_priority_ranking_df(),
            metadata={
                "pipeline_name": "hierarchical_final_decision",
                "created_at_utc": "2026-04-17T12:00:00+00:00",
                "n_rows_input": 3,
                "n_rows_final_decision": 3,
                "n_rows_priority_input": 2,
                "n_rows_priority_ranking": 2,
            },
        ),
    )


def _build_decision_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 101,
                "quality_reason": "pass",
                "review_bucket": "pass",
                "spec_class": "G",
                "spec_subclass": "G2",
                "teff_gspphot": 5750.0,
                "logg_gspphot": 4.35,
                "mh_gspphot": 0.03,
                "bp_rp": 0.82,
                "parallax": 13.2,
                "parallax_over_error": 120.0,
                "ruwe": 0.97,
                "phot_g_mean_mag": 9.2,
                "radius_flame": 1.03,
                "lum_flame": 1.1,
                "evolution_stage": "dwarf",
                "evolstage_flame": "main_sequence",
            },
            {
                "source_id": 102,
                "quality_reason": "pass",
                "review_bucket": "pass",
                "spec_class": "K",
                "spec_subclass": "K1",
                "teff_gspphot": 5050.0,
                "logg_gspphot": 4.45,
                "mh_gspphot": -0.08,
                "bp_rp": 1.02,
                "parallax": 9.4,
                "parallax_over_error": 85.0,
                "ruwe": 1.02,
                "phot_g_mean_mag": 10.8,
                "radius_flame": 0.91,
                "lum_flame": 0.62,
                "evolution_stage": "dwarf",
                "evolstage_flame": "main_sequence",
            },
            {
                "source_id": 103,
                "quality_reason": "review_high_ruwe",
                "review_bucket": "review",
                "spec_class": "F",
                "spec_subclass": "F8",
                "teff_gspphot": 6200.0,
                "logg_gspphot": 4.10,
                "mh_gspphot": -0.02,
                "bp_rp": 0.55,
                "parallax": 3.7,
                "parallax_over_error": 22.0,
                "ruwe": 1.71,
                "phot_g_mean_mag": 11.5,
                "radius_flame": 1.34,
                "lum_flame": 2.4,
                "evolution_stage": "subgiant",
                "evolstage_flame": "subgiant",
            },
        ]
    )


def _build_final_decision_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 101,
                "final_domain_state": "id",
                "final_quality_state": "pass",
                "final_coarse_class": "G",
                "final_refinement_label": "G2",
                "final_refinement_state": "accepted",
                "final_decision_reason": "refinement_accepted",
            },
            {
                "source_id": 102,
                "final_domain_state": "id",
                "final_quality_state": "pass",
                "final_coarse_class": "K",
                "final_refinement_label": "K1",
                "final_refinement_state": "accepted",
                "final_decision_reason": "refinement_accepted",
            },
            {
                "source_id": 103,
                "final_domain_state": "unknown",
                "final_quality_state": "unknown",
                "final_coarse_class": "F",
                "final_refinement_label": pd.NA,
                "final_refinement_state": "blocked",
                "final_decision_reason": "quality_unknown",
            },
        ]
    )


def _build_priority_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 101,
                "spec_class": "G",
                "host_similarity_score": 0.98,
                "observability_score": 0.81,
            },
            {
                "source_id": 102,
                "spec_class": "K",
                "host_similarity_score": 0.92,
                "observability_score": 0.73,
            },
        ]
    )


def _build_priority_ranking_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 101,
                "spec_class": "G",
                "class_priority_score": 0.84,
                "host_similarity_score": 0.98,
                "priority_score": 0.91,
                "priority_label": "high",
                "priority_reason": "host_like_and_observable",
                "observability_score": 0.81,
            },
            {
                "source_id": 102,
                "spec_class": "K",
                "class_priority_score": 0.76,
                "host_similarity_score": 0.92,
                "priority_score": 0.67,
                "priority_label": "medium",
                "priority_reason": "host_like",
                "observability_score": 0.73,
            },
        ]
    )
