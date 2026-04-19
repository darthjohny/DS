# Файл `run_browser.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - визуальный вывод страницы просмотра готового запуска;
# - таблицы распределений и preview текущей выборки без бизнес-логики.
#
# Следующий слой:
# - page-модуль просмотра запуска;
# - helper-модуль подготовки read-only dataframe.

from __future__ import annotations

import pandas as pd
import streamlit as st

from exohost.ui.run_browser_preview import (
    build_ui_run_browser_preview_display_frame,
    build_ui_run_browser_preview_selection_default,
    resolve_ui_run_browser_selected_source_id,
)

from .overview_metrics import render_run_overview_metrics


def render_run_browser(
    *,
    overview,
    domain_distribution_df: pd.DataFrame,
    priority_distribution_df: pd.DataFrame,
    preview_df: pd.DataFrame,
    filtered_row_count: int,
    selected_source_id: str | int | None,
) -> str | None:
    # Страница запуска должна показывать сводку run и результат текущих фильтров отдельно.
    render_run_overview_metrics(overview)
    st.caption(
        f"После текущих фильтров осталось `{filtered_row_count:,}` строк.".replace(",", " ")
    )

    distribution_columns = st.columns(2)
    with distribution_columns[0]:
        st.subheader("Итоговые состояния")
        _render_distribution_table(
            _build_distribution_display_frame(
                domain_distribution_df.rename(
                    columns={
                        "final_domain_state": "Итоговое состояние",
                        "n_rows": "Число строк",
                        "share": "Доля",
                    }
                )
            )
        )

    with distribution_columns[1]:
        st.subheader("Приоритет наблюдений")
        _render_distribution_table(
            _build_distribution_display_frame(
                priority_distribution_df.rename(
                    columns={
                        "priority_label": "Приоритет",
                        "n_rows": "Число строк",
                        "share": "Доля",
                    }
                )
            )
        )

    st.subheader("Preview текущей выборки")
    if preview_df.empty:
        st.info("Для текущих фильтров нет строк в preview.")
        return None

    st.caption(
        "Выделите строку в preview, чтобы синхронизировать `source_id` с блоком "
        "перехода в карточку объекта."
    )
    preview_display_df = build_ui_run_browser_preview_display_frame(preview_df)
    preview_selection_state = st.dataframe(
        preview_display_df,
        width="stretch",
        height="content",
        hide_index=True,
        column_config=_build_preview_column_config(),
        key="run_browser_preview_table",
        on_select="rerun",
        selection_mode="single-row",
        selection_default=build_ui_run_browser_preview_selection_default(
            preview_df,
            selected_source_id=selected_source_id,
        ),
    )
    return resolve_ui_run_browser_selected_source_id(preview_df, preview_selection_state)


def _build_distribution_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Для UI долю удобнее показывать в процентах, а не в raw-дробях.
    if df.empty:
        return df.copy()

    display_df = df.copy()
    if "Доля" in display_df.columns:
        display_df["Доля, %"] = pd.to_numeric(
            display_df["Доля"],
            errors="coerce",
        ) * 100.0
        display_df = display_df.drop(columns=["Доля"])
    return display_df


def _build_preview_column_config() -> dict[str, object]:
    # Column config держим рядом с компонентом, потому что это чисто visual-слой.
    return {
        "source_id": st.column_config.TextColumn(
            "source_id",
            width="medium",
            pinned=True,
        ),
        "Итоговое состояние": st.column_config.TextColumn(
            "Итоговое состояние",
            width="small",
        ),
        "Корзина проверки": st.column_config.TextColumn(
            "Корзина проверки",
            width="small",
        ),
        "Класс": st.column_config.TextColumn("Класс", width="small"),
        "Подкласс": st.column_config.TextColumn("Подкласс", width="small"),
        "Итоговый coarse-класс": st.column_config.TextColumn(
            "Итоговый coarse-класс",
            width="small",
        ),
        "Итоговый подкласс": st.column_config.TextColumn(
            "Итоговый подкласс",
            width="small",
        ),
        "Приоритет": st.column_config.TextColumn("Приоритет", width="small"),
        "Итоговый приоритет": st.column_config.NumberColumn(
            "Итоговый приоритет",
            width="small",
            format="%.3f",
        ),
        "Сходство с host": st.column_config.NumberColumn(
            "Сходство с host",
            width="small",
            format="%.3f",
        ),
        "Наблюдаемость": st.column_config.NumberColumn(
            "Наблюдаемость",
            width="small",
            format="%.3f",
        ),
        "Причина приоритета": st.column_config.TextColumn(
            "Причина приоритета",
            width="large",
        ),
    }


def _build_distribution_column_config() -> dict[str, object]:
    return {
        "Число строк": st.column_config.NumberColumn(
            "Число строк",
            width="small",
            format="%d",
        ),
        "Доля, %": st.column_config.NumberColumn(
            "Доля, %",
            width="small",
            format="%.1f",
        ),
    }


def _render_distribution_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Для этого блока нет доступных строк.")
        return
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config=_build_distribution_column_config(),
    )


__all__ = ["render_run_browser"]
