# Файл `run_browser_page.py` слоя `ui/pages`.
#
# Этот файл отвечает только за:
# - просмотр готового `run_dir`;
# - краткие сводки по артефактам и верхний список кандидатов.
#
# Следующий слой:
# - loader-модули чтения run artifacts;
# - таблицы и компоненты summary-контуров.

from __future__ import annotations

import streamlit as st

from exohost.ui.candidate_card import build_ui_candidate_source_options
from exohost.ui.components.run_browser import render_run_browser
from exohost.ui.components.run_browser_filters import (
    render_run_browser_export,
    render_run_browser_filters,
)
from exohost.ui.components.run_selector import render_run_dir_selector
from exohost.ui.components.source_selector import render_source_id_selector
from exohost.ui.loaders import list_available_run_dirs, load_ui_run_bundle
from exohost.ui.run_browser import build_ui_run_browser_frame
from exohost.ui.run_browser_filters import (
    apply_ui_run_browser_filters,
    build_ui_filtered_domain_distribution_frame,
    build_ui_filtered_priority_distribution_frame,
    build_ui_run_browser_filter_options,
    build_ui_run_browser_preview_frame,
)
from exohost.ui.run_browser_preview import build_ui_run_browser_source_id_options
from exohost.ui.run_overview import build_ui_run_overview
from exohost.ui.session_state import (
    remember_selected_run_dir,
    remember_selected_source_id,
    set_run_load_error,
)


def render_run_browser_page() -> None:
    # Read-only страница остается главным способом просмотра уже собранного artifact bundle.
    st.title("Просмотр запуска")
    st.caption(
        "Страница показывает готовый `run_dir`, итоговые распределения и верхний "
        "список кандидатов без повторного запуска пайплайна."
    )
    available_run_dirs = list_available_run_dirs()
    if not available_run_dirs:
        st.warning("Не удалось найти готовые `run_dir` в `artifacts/decisions`.")
        return

    selected_run_dir = render_run_dir_selector(
        label="Готовый запуск",
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
    run_browser_df = build_ui_run_browser_frame(loaded_bundle)
    filter_options = build_ui_run_browser_filter_options(run_browser_df)
    filters = render_run_browser_filters(filter_options)
    filtered_df = apply_ui_run_browser_filters(run_browser_df, filters)
    selected_preview_source_id = render_run_browser(
        overview=build_ui_run_overview(loaded_bundle),
        domain_distribution_df=build_ui_filtered_domain_distribution_frame(filtered_df),
        priority_distribution_df=build_ui_filtered_priority_distribution_frame(filtered_df),
        preview_df=build_ui_run_browser_preview_frame(
            filtered_df,
            top_n=filters.top_n,
        ),
        filtered_row_count=int(filtered_df.shape[0]),
        selected_source_id=st.session_state.get("selected_source_id"),
    )
    if selected_preview_source_id is not None:
        remember_selected_source_id(st.session_state, selected_preview_source_id)

    render_run_browser_export(
        filtered_df,
        run_dir_name=loaded_bundle.run_dir.name,
    )

    source_id_options = build_ui_run_browser_source_id_options(filtered_df)
    if not source_id_options:
        source_id_options = build_ui_candidate_source_options(loaded_bundle)
    if not source_id_options:
        return

    selected_source_id = render_source_id_selector(
        label="Открыть объект на отдельной странице",
        source_id_options=source_id_options,
        selected_value=st.session_state.get("selected_source_id"),
    )
    remember_selected_source_id(st.session_state, selected_source_id)
    st.caption(
        "Выбор строки в предпросмотре и список ниже используют общий `selected_source_id`. "
        "Детальная карточка доступна на странице «Объект»."
    )
