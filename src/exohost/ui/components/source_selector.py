# Файл `source_selector.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - единый визуальный selectbox выбора `source_id`;
# - восстановление выбранного объекта из session_state.
#
# Следующий слой:
# - страницы `run_browser` и `candidate`;
# - unit-тесты UI component-helper слоя.

from __future__ import annotations

from typing import cast

import streamlit as st


def resolve_ui_selected_source_id_index(
    *,
    options: tuple[str, ...],
    selected_value: object,
) -> int:
    """Возвращает index сохраненного `source_id` для selectbox.

    Сравнение идет через строковое представление, потому что Gaia `source_id`
    может приходить как число из dataframe или как строка из session_state.
    """
    if not options:
        return 0
    selected_key = str(selected_value) if selected_value is not None else None
    if selected_key is None:
        return 0
    try:
        return options.index(selected_key)
    except ValueError:
        return 0


def render_source_id_selector(
    *,
    label: str,
    source_id_options: tuple[str, ...],
    selected_value: object,
) -> str:
    """Рендерит единый selectbox выбора объекта по `source_id`.

    Возвращает выбранный `source_id` строкой, чтобы page-слой мог синхронно
    сохранить его в session_state и передать helper-модулям карточки.
    """
    return cast(
        str,
        st.selectbox(
            label,
            options=source_id_options,
            index=resolve_ui_selected_source_id_index(
                options=source_id_options,
                selected_value=selected_value,
            ),
        ),
    )


__all__ = [
    "render_source_id_selector",
    "resolve_ui_selected_source_id_index",
]
