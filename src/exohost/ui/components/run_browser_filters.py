# Файл `run_browser_filters.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - visual control-panel фильтров страницы запуска;
# - visual export-кнопку для текущей отфильтрованной выборки.
#
# Следующий слой:
# - page-модуль `run_browser_page`;
# - helper-слой `run_browser_filters`.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from exohost.ui.run_browser_filters import (
    UiRunBrowserFilterOptions,
    UiRunBrowserFilters,
    build_ui_run_browser_export_bytes,
)


def render_run_browser_filters(
    filter_options: UiRunBrowserFilterOptions,
    *,
    default_top_n: int = 25,
) -> UiRunBrowserFilters:
    # Виджеты фильтрации держим в отдельном компоненте, чтобы page-слой не знал деталей UI controls.
    with st.expander("Фильтры", expanded=True):
        first_row = st.columns(4)
        selected_domain_states = tuple(
            first_row[0].multiselect(
                "Итоговое состояние",
                options=filter_options.final_domain_states,
                default=(),
                placeholder="Все состояния",
            )
        )
        selected_priority_labels = tuple(
            first_row[1].multiselect(
                "Приоритет",
                options=filter_options.priority_labels,
                default=(),
                placeholder="Все приоритеты",
            )
        )
        selected_spec_classes = tuple(
            first_row[2].multiselect(
                "Спектральный класс",
                options=filter_options.spec_classes,
                default=(),
                placeholder="Все классы",
            )
        )
        source_id_query = first_row[3].text_input(
            "Поиск по source_id",
            value="",
            placeholder="Например, 3618253627527608832",
            help=(
                "Поиск идет по строковому представлению `source_id`, "
                "поэтому подходит и для очень больших идентификаторов."
            ),
        )

        second_row = st.columns((1, 3))
        top_n_value = int(
            second_row[0].number_input(
                "Строк в предпросмотре",
                min_value=1,
                value=max(1, int(default_top_n)),
                step=5,
            )
        )
        second_row[1].caption(
            "Пустой multiselect означает, что соответствующий фильтр не применяется."
        )

    return UiRunBrowserFilters(
        final_domain_states=selected_domain_states,
        priority_labels=selected_priority_labels,
        spec_classes=selected_spec_classes,
        source_id_query=source_id_query,
        top_n=top_n_value,
    )


def render_run_browser_export(
    filtered_df: pd.DataFrame,
    *,
    run_dir_name: str,
) -> None:
    # Export должен работать от текущей filtered selection и не заставлять страницу rerun без нужды.
    st.subheader("Экспорт выборки")
    if filtered_df.empty:
        st.info("Для текущего набора фильтров нет строк для выгрузки.")
        return

    st.download_button(
        "Скачать текущую выборку CSV",
        data=build_ui_run_browser_export_bytes(filtered_df),
        file_name=f"{Path(run_dir_name).name}_filtered_selection.csv",
        mime="text/csv",
        on_click="ignore",
        width="stretch",
    )


__all__ = [
    "render_run_browser_export",
    "render_run_browser_filters",
]
