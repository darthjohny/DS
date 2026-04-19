# Тестовый файл `test_ui_candidate_card.py` домена `ui`.
#
# Этот файл проверяет только:
# - helper-слой карточки объекта;
# - поиск по `source_id` и разбор route/physics по одной звезде.
#
# Следующий слой:
# - компонент карточки объекта;
# - страница выбора и просмотра одной звезды.

from __future__ import annotations

from exohost.ui.candidate_card import (
    build_ui_candidate_physics_frame,
    build_ui_candidate_source_options,
    build_ui_candidate_summary_frame,
)

from .ui_testkit import build_ui_loaded_run_bundle


def test_build_ui_candidate_source_options_returns_all_available_source_ids() -> None:
    source_options = build_ui_candidate_source_options(build_ui_loaded_run_bundle())

    assert source_options == ("101", "102", "103")


def test_build_ui_candidate_summary_frame_returns_pipeline_route_for_source_id() -> None:
    summary_df = build_ui_candidate_summary_frame(build_ui_loaded_run_bundle(), "101")

    assert list(summary_df["source_id"]) == [101]
    assert list(summary_df["final_domain_state"].astype(str)) == ["id"]
    assert list(summary_df["priority_label"].astype(str)) == ["high"]


def test_build_ui_candidate_physics_frame_returns_physical_preview_for_source_id() -> None:
    physics_df = build_ui_candidate_physics_frame(build_ui_loaded_run_bundle(), 102)

    assert list(physics_df["source_id"]) == [102]
    assert list(physics_df["spec_subclass"].astype(str)) == ["K1"]
    assert float(physics_df.loc[0, "teff_gspphot"]) == 5050.0


def test_build_ui_candidate_summary_frame_returns_empty_frame_for_unknown_source_id() -> None:
    summary_df = build_ui_candidate_summary_frame(build_ui_loaded_run_bundle(), "999")

    assert summary_df.empty
