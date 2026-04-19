# Тестовый файл `test_ui_home_summary.py` домена `ui`.
#
# Этот файл проверяет только:
# - helper-слой главной страницы интерфейса;
# - сборку главного прикладного результата и поиск локальной схемы системы.
#
# Следующий слой:
# - визуальные компоненты домашней страницы;
# - page-level smoke главной страницы Streamlit.

from __future__ import annotations

from pathlib import Path

from exohost.ui.home_summary import (
    build_ui_home_main_result,
    build_ui_home_top_candidates_preview,
    resolve_ui_system_overview_path,
)

from .ui_testkit import build_ui_loaded_run_bundle


def test_build_ui_home_main_result_extracts_shortlist_counts() -> None:
    main_result = build_ui_home_main_result(build_ui_loaded_run_bundle())

    assert main_result.run_dir_name == "hierarchical_final_decision_demo"
    assert main_result.total_objects == 3
    assert main_result.ranked_count == 2
    assert main_result.high_priority_count == 1
    assert main_result.medium_priority_count == 1
    assert main_result.low_priority_count == 0
    assert main_result.high_priority_share == 0.5


def test_build_ui_home_top_candidates_preview_limits_and_keeps_order() -> None:
    preview_df = build_ui_home_top_candidates_preview(
        build_ui_loaded_run_bundle(),
        top_n=1,
    )

    assert list(preview_df["source_id"]) == [101]
    assert list(preview_df["priority_label"].astype(str)) == ["high"]


def test_resolve_ui_system_overview_path_returns_first_existing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.svg"
    second_path = tmp_path / "system_overview_ru.svg"
    second_path.write_text("<svg></svg>", encoding="utf-8")

    resolved_path = resolve_ui_system_overview_path((missing_path, second_path))

    assert resolved_path == second_path.resolve()


def test_resolve_ui_system_overview_path_returns_none_when_nothing_exists(
    tmp_path: Path,
) -> None:
    resolved_path = resolve_ui_system_overview_path((tmp_path / "a.svg", tmp_path / "b.svg"))

    assert resolved_path is None
