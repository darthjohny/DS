# Файл `run_browser_preview.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - display-contract preview-таблицы страницы просмотра запуска;
# - безопасную работу с выбором строки `st.dataframe` и `source_id`.
#
# Следующий слой:
# - visual component preview-таблицы;
# - unit-тесты helper-слоя run-browser preview.

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import TypedDict

import pandas as pd

RUN_BROWSER_PREVIEW_DISPLAY_COLUMNS: tuple[str, ...] = (
    "source_id",
    "final_domain_state",
    "review_bucket",
    "spec_class",
    "spec_subclass",
    "final_coarse_class",
    "final_refinement_label",
    "priority_label",
    "priority_score",
    "host_similarity_score",
    "observability_score",
    "priority_reason",
)

RUN_BROWSER_PREVIEW_DISPLAY_LABELS: dict[str, str] = {
    "source_id": "source_id",
    "final_domain_state": "Итоговое состояние",
    "review_bucket": "Корзина проверки",
    "spec_class": "Класс",
    "spec_subclass": "Подкласс",
    "final_coarse_class": "Итоговый крупный класс",
    "final_refinement_label": "Итоговый подкласс",
    "priority_label": "Приоритет",
    "priority_score": "Итоговый приоритет",
    "host_similarity_score": "Сходство со звездами-хозяевами",
    "observability_score": "Наблюдаемость",
    "priority_reason": "Причина приоритета",
}


class UiDataframeSelectionState(TypedDict):
    # Тип соответствует documented schema `st.dataframe(..., on_select=...)`.
    rows: list[int]
    columns: list[str]
    cells: list[tuple[int, str]]


class UiDataframeState(TypedDict):
    # Streamlit возвращает словарь с вложенным selection-state.
    selection: UiDataframeSelectionState


def build_ui_run_browser_preview_display_frame(preview_df: pd.DataFrame) -> pd.DataFrame:
    # Preview-таблица должна работать от стабильного набора колонок и их порядка.
    display_df = preview_df.copy()
    for column_name in RUN_BROWSER_PREVIEW_DISPLAY_COLUMNS:
        if column_name not in display_df.columns:
            display_df[column_name] = pd.NA
    return display_df.loc[:, list(RUN_BROWSER_PREVIEW_DISPLAY_COLUMNS)].rename(
        columns=RUN_BROWSER_PREVIEW_DISPLAY_LABELS
    )


def build_ui_run_browser_source_id_options(filtered_df: pd.DataFrame) -> tuple[str, ...]:
    # Selectbox карточки объекта должен получать уже готовый список source_id без page-логики.
    if filtered_df.empty or "source_id" not in filtered_df.columns:
        return ()

    source_ids = (
        filtered_df["source_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return tuple(source_ids)


def build_ui_run_browser_preview_selection_default(
    preview_df: pd.DataFrame,
    *,
    selected_source_id: str | int | None,
) -> UiDataframeState | None:
    # Текущий выбранный объект подсвечиваем в preview, если он виден в таблице.
    if preview_df.empty or selected_source_id is None or "source_id" not in preview_df.columns:
        return None

    source_id_key = _to_optional_string(selected_source_id)
    if source_id_key is None:
        return None

    source_id_values = preview_df["source_id"].astype(str).tolist()
    try:
        selected_row_index = source_id_values.index(source_id_key)
    except ValueError:
        return None

    return {
        "selection": {
            "rows": [selected_row_index],
            "columns": [],
            "cells": [],
        }
    }


def resolve_ui_run_browser_selected_source_id(
    preview_df: pd.DataFrame,
    selection_state: UiDataframeState | Mapping[str, object] | None,
) -> str | None:
    # Переводим selection-state обратно в `source_id`, чтобы page-слой работал только с доменным значением.
    if preview_df.empty or "source_id" not in preview_df.columns:
        return None
    if not isinstance(selection_state, Mapping):
        return None

    raw_selection = selection_state.get("selection")
    if not isinstance(raw_selection, Mapping):
        return None

    raw_rows = raw_selection.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 1:
        return None

    selected_row_index = raw_rows[0]
    if isinstance(selected_row_index, bool) or not isinstance(selected_row_index, Integral):
        return None

    row_position = int(selected_row_index)
    if row_position < 0 or row_position >= len(preview_df):
        return None

    return _to_optional_string(preview_df.iloc[row_position]["source_id"])


def _to_optional_string(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value or None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Real):
        if pd.isna(value):
            return None
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return None


__all__ = [
    "RUN_BROWSER_PREVIEW_DISPLAY_COLUMNS",
    "RUN_BROWSER_PREVIEW_DISPLAY_LABELS",
    "UiDataframeSelectionState",
    "UiDataframeState",
    "build_ui_run_browser_preview_display_frame",
    "build_ui_run_browser_preview_selection_default",
    "build_ui_run_browser_source_id_options",
    "resolve_ui_run_browser_selected_source_id",
]
