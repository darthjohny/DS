# Тестовый файл `test_ui_run_browser_preview.py` домена `ui`.
#
# Этот файл проверяет только:
# - preview-helper страницы просмотра запуска;
# - display-contract таблицы и разбор row-selection в `source_id`.
#
# Следующий слой:
# - visual component preview-таблицы;
# - page-уровень `run_browser` и smoke-проверки Streamlit.

from __future__ import annotations

from exohost.ui.run_browser import build_ui_run_browser_frame
from exohost.ui.run_browser_preview import (
    RUN_BROWSER_PREVIEW_DISPLAY_LABELS,
    build_ui_run_browser_preview_display_frame,
    build_ui_run_browser_preview_selection_default,
    build_ui_run_browser_source_id_options,
    resolve_ui_run_browser_selected_source_id,
)

from .ui_testkit import build_ui_loaded_run_bundle


def test_build_ui_run_browser_preview_display_frame_applies_expected_columns() -> None:
    preview_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle()).head(2)

    display_df = build_ui_run_browser_preview_display_frame(preview_df)

    assert tuple(display_df.columns) == tuple(RUN_BROWSER_PREVIEW_DISPLAY_LABELS.values())
    assert list(display_df["source_id"]) == [101, 102]


def test_build_ui_run_browser_source_id_options_keeps_filtered_order() -> None:
    filtered_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle())

    source_id_options = build_ui_run_browser_source_id_options(filtered_df)

    assert source_id_options == ("101", "102", "103")


def test_build_ui_run_browser_preview_selection_default_uses_current_source_id() -> None:
    preview_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle()).head(2)

    selection_default = build_ui_run_browser_preview_selection_default(
        preview_df,
        selected_source_id="102",
    )

    assert selection_default == {
        "selection": {
            "rows": [1],
            "columns": [],
            "cells": [],
        }
    }


def test_resolve_ui_run_browser_selected_source_id_returns_selected_row_value() -> None:
    preview_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle()).head(2)

    selected_source_id = resolve_ui_run_browser_selected_source_id(
        preview_df,
        {"selection": {"rows": [1], "columns": [], "cells": []}},
    )

    assert selected_source_id == "102"


def test_resolve_ui_run_browser_selected_source_id_ignores_invalid_state() -> None:
    preview_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle()).head(2)

    selected_source_id = resolve_ui_run_browser_selected_source_id(
        preview_df,
        {"selection": {"rows": [99], "columns": [], "cells": []}},
    )

    assert selected_source_id is None
