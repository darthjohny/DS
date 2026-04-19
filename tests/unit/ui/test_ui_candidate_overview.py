# Тестовый файл `test_ui_candidate_overview.py` домена `ui`.
#
# Этот файл проверяет только:
# - compact overview helper для карточки объекта;
# - route-сводку по шагам pipeline для выбранного `source_id`.
#
# Следующий слой:
# - визуальный компонент карточки объекта;
# - smoke-проверки page-рендера и виджетов.

from __future__ import annotations

import pandas as pd

from exohost.ui.candidate_card import (
    build_ui_candidate_physics_frame,
    build_ui_candidate_summary_frame,
)
from exohost.ui.candidate_overview import (
    build_ui_candidate_overview,
    build_ui_candidate_route_frame,
)

from .ui_testkit import build_ui_loaded_run_bundle


def test_build_ui_candidate_overview_returns_compact_summary_for_ranked_object() -> None:
    bundle = build_ui_loaded_run_bundle()
    summary_df = build_ui_candidate_summary_frame(bundle, "101")
    physics_df = build_ui_candidate_physics_frame(bundle, "101")

    overview = build_ui_candidate_overview(summary_df, physics_df)

    assert overview is not None
    assert overview.source_id == "101"
    assert overview.final_domain_state == "id"
    assert overview.final_refinement_label == "G2"
    assert overview.priority_label == "high"
    assert overview.spec_subclass == "G2"
    assert overview.teff_gspphot == 5750.0
    assert "refinement_accepted" in overview.overview_note


def test_build_ui_candidate_route_frame_returns_expected_stage_order() -> None:
    summary_df = build_ui_candidate_summary_frame(build_ui_loaded_run_bundle(), "101")

    route_df = build_ui_candidate_route_frame(summary_df)

    assert list(route_df["stage_name"]) == [
        "Quality gate",
        "Refinement",
        "Final routing",
        "Priority ranking",
    ]
    assert list(route_df["stage_state"].astype(str)) == [
        "pass",
        "accepted",
        "id",
        "high",
    ]


def test_build_ui_candidate_overview_handles_missing_priority_data() -> None:
    bundle = build_ui_loaded_run_bundle()
    summary_df = build_ui_candidate_summary_frame(bundle, "103")
    physics_df = build_ui_candidate_physics_frame(bundle, "103")

    overview = build_ui_candidate_overview(summary_df, physics_df)
    route_df = build_ui_candidate_route_frame(summary_df)

    assert overview is not None
    assert overview.priority_label is None
    assert "не рассчитан priority ranking" in overview.overview_note
    assert pd.isna(
        route_df.loc[
            route_df["stage_name"] == "Priority ranking",
            "stage_state",
        ].iloc[0]
    )


def test_build_ui_candidate_overview_returns_none_for_empty_summary() -> None:
    bundle = build_ui_loaded_run_bundle()
    summary_df = build_ui_candidate_summary_frame(bundle, "999")
    physics_df = build_ui_candidate_physics_frame(bundle, "999")

    overview = build_ui_candidate_overview(summary_df, physics_df)

    assert overview is None
