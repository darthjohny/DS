# Тестовый файл `test_ui_session_state.py` домена `ui`.
#
# Этот файл проверяет только:
# - инициализацию минимального состояния интерфейса;
# - обновление служебных ключей без смешения разных сценариев UI.
#
# Следующий слой:
# - entrypoint и страницы Streamlit;
# - smoke-тест интерфейсного контура.

from __future__ import annotations

from exohost.ui.contracts import UI_SESSION_STATE_CONTRACT
from exohost.ui.session_state import (
    clear_ui_errors,
    initialize_ui_session_state,
    remember_generated_run_dir,
    remember_selected_run_dir,
    remember_selected_source_id,
    remember_uploaded_csv_path,
    set_csv_validation_error,
    set_run_load_error,
)


def test_initialize_ui_session_state_sets_missing_defaults() -> None:
    session_state: dict[str, object] = {}

    initialize_ui_session_state(session_state)

    assert session_state["current_page"] == UI_SESSION_STATE_CONTRACT.default_page
    assert session_state[UI_SESSION_STATE_CONTRACT.selected_run_dir_key] is None
    assert session_state[UI_SESSION_STATE_CONTRACT.selected_source_id_key] is None
    assert session_state[UI_SESSION_STATE_CONTRACT.uploaded_csv_path_key] is None
    assert session_state[UI_SESSION_STATE_CONTRACT.generated_run_dir_key] is None
    assert session_state[UI_SESSION_STATE_CONTRACT.run_load_error_key] is None
    assert session_state[UI_SESSION_STATE_CONTRACT.csv_validation_error_key] is None


def test_initialize_ui_session_state_does_not_overwrite_existing_values() -> None:
    session_state: dict[str, object] = {
        "current_page": "metrics",
        UI_SESSION_STATE_CONTRACT.selected_run_dir_key: "/tmp/run",
    }

    initialize_ui_session_state(session_state)

    assert session_state["current_page"] == "metrics"
    assert session_state[UI_SESSION_STATE_CONTRACT.selected_run_dir_key] == "/tmp/run"


def test_ui_session_state_helpers_update_expected_keys() -> None:
    session_state: dict[str, object] = {}
    initialize_ui_session_state(session_state)

    remember_selected_run_dir(session_state, "/tmp/run_a")
    remember_selected_source_id(session_state, 123456789)
    remember_uploaded_csv_path(session_state, "/tmp/input.csv")
    remember_generated_run_dir(session_state, "/tmp/run_b")
    set_run_load_error(session_state, "run error")
    set_csv_validation_error(session_state, "csv error")

    assert session_state[UI_SESSION_STATE_CONTRACT.selected_run_dir_key] == "/tmp/run_a"
    assert session_state[UI_SESSION_STATE_CONTRACT.selected_source_id_key] == 123456789
    assert session_state[UI_SESSION_STATE_CONTRACT.uploaded_csv_path_key] == "/tmp/input.csv"
    assert session_state[UI_SESSION_STATE_CONTRACT.generated_run_dir_key] == "/tmp/run_b"
    assert session_state[UI_SESSION_STATE_CONTRACT.run_load_error_key] == "run error"
    assert session_state[UI_SESSION_STATE_CONTRACT.csv_validation_error_key] == "csv error"


def test_clear_ui_errors_resets_only_error_keys() -> None:
    session_state: dict[str, object] = {}
    initialize_ui_session_state(session_state)
    remember_selected_run_dir(session_state, "/tmp/run_a")
    set_run_load_error(session_state, "run error")
    set_csv_validation_error(session_state, "csv error")

    clear_ui_errors(session_state)

    assert session_state[UI_SESSION_STATE_CONTRACT.selected_run_dir_key] == "/tmp/run_a"
    assert session_state[UI_SESSION_STATE_CONTRACT.run_load_error_key] is None
    assert session_state[UI_SESSION_STATE_CONTRACT.csv_validation_error_key] is None
