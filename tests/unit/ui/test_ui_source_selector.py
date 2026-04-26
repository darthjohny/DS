# Тестовый файл `test_ui_source_selector.py` домена `ui`.
#
# Этот файл проверяет только:
# - helper-логику общего компонента выбора `source_id`;
# - восстановление выбранного объекта из session_state.
#
# Следующий слой:
# - Streamlit-страницы, использующие общий source selector;
# - smoke-проверка страниц интерфейса.

from __future__ import annotations

from exohost.ui.components.source_selector import resolve_ui_selected_source_id_index


def test_resolve_ui_selected_source_id_index_restores_string_selection() -> None:
    selected_index = resolve_ui_selected_source_id_index(
        options=("101", "102", "103"),
        selected_value="102",
    )

    assert selected_index == 1


def test_resolve_ui_selected_source_id_index_matches_numeric_session_value() -> None:
    selected_index = resolve_ui_selected_source_id_index(
        options=("101", "102", "103"),
        selected_value=103,
    )

    assert selected_index == 2


def test_resolve_ui_selected_source_id_index_falls_back_to_first_option() -> None:
    selected_index = resolve_ui_selected_source_id_index(
        options=("101", "102", "103"),
        selected_value="999",
    )

    assert selected_index == 0


def test_resolve_ui_selected_source_id_index_handles_empty_options() -> None:
    selected_index = resolve_ui_selected_source_id_index(
        options=(),
        selected_value="101",
    )

    assert selected_index == 0
