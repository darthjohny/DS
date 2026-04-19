# Файл `session_state.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - инициализацию и обновление минимального состояния интерфейса;
# - единые ключи и безопасную работу со служебными значениями UI.
#
# Следующий слой:
# - entrypoint и страницы Streamlit;
# - unit-тесты state-helper слоя.

from __future__ import annotations

from typing import Any, Protocol

from exohost.ui.contracts import UI_SESSION_STATE_CONTRACT


class UiSessionStateStore(Protocol):
    # Streamlit дает proxy-объект, а тесты работают с обычным dict.
    # Для helper-слоя достаточно зафиксировать только реально используемые операции.
    def __contains__(self, key: object, /) -> bool: ...

    def __getitem__(self, key: str, /) -> Any: ...

    def __setitem__(self, key: str, value: Any, /) -> None: ...


def initialize_ui_session_state(session_state: UiSessionStateStore) -> None:
    # Заполняем только недостающие ключи, чтобы повторный rerun не терял выбор пользователя.
    _set_state_default(
        session_state,
        key="current_page",
        value=UI_SESSION_STATE_CONTRACT.default_page,
    )
    _set_state_default(
        session_state,
        key=UI_SESSION_STATE_CONTRACT.selected_run_dir_key,
        value=None,
    )
    _set_state_default(
        session_state,
        key=UI_SESSION_STATE_CONTRACT.selected_source_id_key,
        value=None,
    )
    _set_state_default(
        session_state,
        key=UI_SESSION_STATE_CONTRACT.uploaded_csv_path_key,
        value=None,
    )
    _set_state_default(
        session_state,
        key=UI_SESSION_STATE_CONTRACT.generated_run_dir_key,
        value=None,
    )
    _set_state_default(
        session_state,
        key=UI_SESSION_STATE_CONTRACT.run_load_error_key,
        value=None,
    )
    _set_state_default(
        session_state,
        key=UI_SESSION_STATE_CONTRACT.csv_validation_error_key,
        value=None,
    )


def remember_selected_run_dir(
    session_state: UiSessionStateStore,
    run_dir: str | None,
) -> None:
    # Выбранный run_dir используется как главный anchor для read-only страниц интерфейса.
    session_state[UI_SESSION_STATE_CONTRACT.selected_run_dir_key] = run_dir


def remember_selected_source_id(
    session_state: UiSessionStateStore,
    source_id: str | int | None,
) -> None:
    # Карточка объекта должна жить отдельно от таблицы, поэтому source_id храним явно.
    session_state[UI_SESSION_STATE_CONTRACT.selected_source_id_key] = source_id


def remember_uploaded_csv_path(
    session_state: UiSessionStateStore,
    csv_path: str | None,
) -> None:
    # Храним только путь к временному CSV, а не содержимое файла внутри session_state.
    session_state[UI_SESSION_STATE_CONTRACT.uploaded_csv_path_key] = csv_path


def remember_generated_run_dir(
    session_state: UiSessionStateStore,
    run_dir: str | None,
) -> None:
    # Новый run_dir нужен для быстрого перехода от кнопочного запуска к просмотру результатов.
    session_state[UI_SESSION_STATE_CONTRACT.generated_run_dir_key] = run_dir


def set_run_load_error(
    session_state: UiSessionStateStore,
    error_message: str | None,
) -> None:
    # Отдельно храним ошибку чтения run_dir, чтобы не смешивать ее с ошибками CSV.
    session_state[UI_SESSION_STATE_CONTRACT.run_load_error_key] = error_message


def set_csv_validation_error(
    session_state: UiSessionStateStore,
    error_message: str | None,
) -> None:
    # Ошибка внешнего CSV — это отдельный служебный сигнал для страницы запуска.
    session_state[UI_SESSION_STATE_CONTRACT.csv_validation_error_key] = error_message


def clear_ui_errors(session_state: UiSessionStateStore) -> None:
    # Перед новым чтением run_dir или новым запуском очищаем обе служебные ошибки разом.
    set_run_load_error(session_state, None)
    set_csv_validation_error(session_state, None)


def _set_state_default(
    session_state: UiSessionStateStore,
    *,
    key: str,
    value: Any,
) -> None:
    if key in session_state:
        return
    session_state[key] = value


__all__ = [
    "clear_ui_errors",
    "initialize_ui_session_state",
    "remember_generated_run_dir",
    "remember_selected_run_dir",
    "remember_selected_source_id",
    "remember_uploaded_csv_path",
    "set_csv_validation_error",
    "set_run_load_error",
    "UiSessionStateStore",
]
