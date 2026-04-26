# Файл `run_selector.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - единый визуальный selectbox выбора готового `run_dir`;
# - маленькие pure-helper функции для стабильного восстановления выбора.
#
# Следующий слой:
# - страницы `run_browser`, `candidate` и `csv_decide`;
# - unit-тесты UI component-helper слоя.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import streamlit as st


def build_ui_run_dir_options(run_dirs: Sequence[str | Path]) -> tuple[str, ...]:
    """Собирает строковые options для Streamlit selectbox.

    Функция принимает найденные `run_dir` как `Path` или строки и возвращает
    стабильный tuple строк, который можно сравнивать с session_state.
    """
    return tuple(str(path) for path in run_dirs)


def resolve_ui_selected_run_dir_index(
    *,
    options: tuple[str, ...],
    selected_value: object,
) -> int:
    """Возвращает index сохраненного `run_dir` для selectbox.

    Если сохраненного значения нет или оно больше не входит в options, функция
    безопасно выбирает первый доступный запуск.
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


def format_ui_run_dir_option(value: str) -> str:
    """Форматирует путь `run_dir` для компактного показа в selectbox."""
    return Path(value).name


def render_run_dir_selector(
    *,
    label: str,
    run_dirs: Sequence[str | Path],
    selected_value: object,
) -> str:
    """Рендерит единый selectbox выбора готового `run_dir`.

    Возвращает выбранный путь строкой, чтобы page-слой мог сохранить его в
    session_state и загрузить bundle через loader-слой.
    """
    run_dir_options = build_ui_run_dir_options(run_dirs)
    return cast(
        str,
        st.selectbox(
            label,
            options=run_dir_options,
            index=resolve_ui_selected_run_dir_index(
                options=run_dir_options,
                selected_value=selected_value,
            ),
            format_func=format_ui_run_dir_option,
        ),
    )


__all__ = [
    "build_ui_run_dir_options",
    "format_ui_run_dir_option",
    "render_run_dir_selector",
    "resolve_ui_selected_run_dir_index",
]
