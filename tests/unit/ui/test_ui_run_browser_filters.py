# Тестовый файл `test_ui_run_browser_filters.py` домена `ui`.
#
# Этот файл проверяет только:
# - helper-логику фильтрации страницы просмотра запуска;
# - export и распределения по текущей отфильтрованной выборке.
#
# Следующий слой:
# - visual control-panel фильтров и выгрузки;
# - page-level smoke страницы `run-browser`.

from __future__ import annotations

from exohost.ui.run_browser import build_ui_run_browser_frame
from exohost.ui.run_browser_filters import (
    UiRunBrowserFilters,
    apply_ui_run_browser_filters,
    build_ui_filtered_domain_distribution_frame,
    build_ui_filtered_priority_distribution_frame,
    build_ui_run_browser_export_bytes,
    build_ui_run_browser_filter_options,
    build_ui_run_browser_preview_frame,
)

from .ui_testkit import build_ui_loaded_run_bundle


def test_build_ui_run_browser_filter_options_collects_expected_labels() -> None:
    filter_options = build_ui_run_browser_filter_options(
        build_ui_run_browser_frame(build_ui_loaded_run_bundle())
    )

    assert filter_options.final_domain_states == ("id", "unknown")
    assert filter_options.priority_labels == ("high", "medium")
    assert filter_options.spec_classes == ("F", "G", "K")


def test_apply_ui_run_browser_filters_keeps_only_matching_ranked_rows() -> None:
    run_browser_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle())

    filtered_df = apply_ui_run_browser_filters(
        run_browser_df,
        UiRunBrowserFilters(
            final_domain_states=("id",),
            priority_labels=("high",),
            spec_classes=("G",),
        ),
    )

    assert list(filtered_df["source_id"]) == [101]
    assert list(filtered_df["priority_label"].astype(str)) == ["high"]


def test_apply_ui_run_browser_filters_supports_source_id_text_query() -> None:
    run_browser_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle())

    filtered_df = apply_ui_run_browser_filters(
        run_browser_df,
        UiRunBrowserFilters(source_id_query="03"),
    )

    assert list(filtered_df["source_id"]) == [103]


def test_build_ui_filtered_priority_distribution_frame_drops_unranked_rows() -> None:
    run_browser_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle())

    distribution_df = build_ui_filtered_priority_distribution_frame(run_browser_df)

    assert set(distribution_df["priority_label"].astype(str)) == {"high", "medium"}
    assert float(distribution_df["share"].sum()) == 1.0


def test_build_ui_run_browser_preview_frame_limits_rows() -> None:
    run_browser_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle())

    preview_df = build_ui_run_browser_preview_frame(run_browser_df, top_n=1)

    assert list(preview_df["source_id"]) == [101]


def test_build_ui_run_browser_export_bytes_serializes_current_selection() -> None:
    run_browser_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle())
    filtered_df = apply_ui_run_browser_filters(
        run_browser_df,
        UiRunBrowserFilters(priority_labels=("medium",)),
    )

    payload = build_ui_run_browser_export_bytes(filtered_df).decode("utf-8")

    assert "source_id" in payload
    assert "102" in payload


def test_build_ui_filtered_domain_distribution_frame_counts_filtered_rows() -> None:
    run_browser_df = build_ui_run_browser_frame(build_ui_loaded_run_bundle())
    filtered_df = apply_ui_run_browser_filters(
        run_browser_df,
        UiRunBrowserFilters(final_domain_states=("unknown",)),
    )

    distribution_df = build_ui_filtered_domain_distribution_frame(filtered_df)

    assert list(distribution_df["final_domain_state"].astype(str)) == ["unknown"]
    assert int(distribution_df.loc[0, "n_rows"]) == 1
