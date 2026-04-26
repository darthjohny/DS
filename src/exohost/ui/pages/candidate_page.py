# Файл `candidate_page.py` слоя `ui/pages`.
#
# Этот файл отвечает только за:
# - карточку одной звезды;
# - показ итогового решения, причин и ключевых физических параметров.
#
# Следующий слой:
# - loader-слой объекта и summary-компоненты;
# - связка с таблицей кандидатов и `source_id`.

from __future__ import annotations

import streamlit as st

from exohost.ui.candidate_card import (
    build_ui_candidate_physics_frame,
    build_ui_candidate_source_options,
    build_ui_candidate_summary_frame,
)
from exohost.ui.candidate_overview import (
    build_ui_candidate_overview,
    build_ui_candidate_route_frame,
)
from exohost.ui.components.candidate_card import render_candidate_card
from exohost.ui.components.run_selector import render_run_dir_selector
from exohost.ui.components.source_selector import render_source_id_selector
from exohost.ui.loaders import list_available_run_dirs, load_ui_run_bundle
from exohost.ui.session_state import (
    remember_selected_run_dir,
    remember_selected_source_id,
    set_run_load_error,
)


def render_candidate_page() -> None:
    # Детальная карточка живет отдельно, чтобы не перегружать общую страницу запуска.
    st.title("Карточка объекта")
    st.caption(
        "Страница собирает маршрут по пайплайну и физические параметры одной "
        "звезды из уже готового `run_dir`."
    )
    available_run_dirs = list_available_run_dirs()
    if not available_run_dirs:
        st.warning("Не удалось найти готовые `run_dir` в `artifacts/decisions`.")
        return

    selected_run_dir = render_run_dir_selector(
        label="Источник данных карточки",
        run_dirs=available_run_dirs,
        selected_value=st.session_state.get("selected_run_dir"),
    )
    remember_selected_run_dir(st.session_state, selected_run_dir)

    try:
        loaded_bundle = load_ui_run_bundle(selected_run_dir)
    except RuntimeError as exc:
        set_run_load_error(st.session_state, str(exc))
        st.error(f"Не удалось загрузить выбранный запуск: {exc}")
        return

    set_run_load_error(st.session_state, None)
    source_id_options = build_ui_candidate_source_options(loaded_bundle)
    if not source_id_options:
        st.warning("В выбранном запуске нет доступных `source_id` для карточки.")
        return

    selected_source_id = render_source_id_selector(
        label="Объект (`source_id`)",
        source_id_options=source_id_options,
        selected_value=st.session_state.get("selected_source_id"),
    )
    remember_selected_source_id(st.session_state, selected_source_id)

    summary_df = build_ui_candidate_summary_frame(loaded_bundle, selected_source_id)
    physics_df = build_ui_candidate_physics_frame(loaded_bundle, selected_source_id)
    overview = build_ui_candidate_overview(summary_df, physics_df)
    if overview is None:
        st.warning("Не удалось собрать overview карточки для выбранного `source_id`.")
        return

    render_candidate_card(
        overview=overview,
        route_df=build_ui_candidate_route_frame(summary_df),
        summary_df=summary_df,
        physics_df=physics_df,
    )
