# Тестовый файл `test_ui_run_selector.py` домена `ui`.
#
# Этот файл проверяет только:
# - helper-логику общего компонента выбора `run_dir`;
# - стабильное восстановление выбора из session_state.
#
# Следующий слой:
# - Streamlit-страницы, использующие общий run selector;
# - smoke-проверка страниц интерфейса.

from __future__ import annotations

from pathlib import Path

from exohost.ui.components.run_selector import (
    build_ui_run_dir_options,
    format_ui_run_dir_option,
    resolve_ui_selected_run_dir_index,
)


def test_build_ui_run_dir_options_normalizes_paths_to_strings() -> None:
    options = build_ui_run_dir_options(
        [
            Path("artifacts/decisions/run_a"),
            "artifacts/decisions/run_b",
        ]
    )

    assert options == (
        "artifacts/decisions/run_a",
        "artifacts/decisions/run_b",
    )


def test_resolve_ui_selected_run_dir_index_restores_known_selection() -> None:
    options = (
        "artifacts/decisions/run_a",
        "artifacts/decisions/run_b",
    )

    selected_index = resolve_ui_selected_run_dir_index(
        options=options,
        selected_value="artifacts/decisions/run_b",
    )

    assert selected_index == 1


def test_resolve_ui_selected_run_dir_index_falls_back_to_first_option() -> None:
    options = (
        "artifacts/decisions/run_a",
        "artifacts/decisions/run_b",
    )

    selected_index = resolve_ui_selected_run_dir_index(
        options=options,
        selected_value="artifacts/decisions/missing_run",
    )

    assert selected_index == 0


def test_format_ui_run_dir_option_shows_compact_name() -> None:
    label = format_ui_run_dir_option(
        "/tmp/artifacts/decisions/hierarchical_final_decision_demo"
    )

    assert label == "hierarchical_final_decision_demo"
