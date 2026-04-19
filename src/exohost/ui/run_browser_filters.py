# Файл `run_browser_filters.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - helper-логику фильтрации merged-таблицы страницы запуска;
# - подготовку export-CSV и компактных распределений после фильтрации.
#
# Следующий слой:
# - Streamlit-компонент control-panel для фильтров и выгрузки;
# - unit-тесты filter-helper слоя.

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class UiRunBrowserFilters:
    # Фильтры страницы запуска держим как отдельный immutable helper-контракт.
    final_domain_states: tuple[str, ...] = ()
    priority_labels: tuple[str, ...] = ()
    spec_classes: tuple[str, ...] = ()
    source_id_query: str | None = None
    top_n: int = 25


@dataclass(frozen=True, slots=True)
class UiRunBrowserFilterOptions:
    # UI должен получать уже готовые списки значений, а не собирать их в page-слое.
    final_domain_states: tuple[str, ...]
    priority_labels: tuple[str, ...]
    spec_classes: tuple[str, ...]


def build_ui_run_browser_filter_options(run_browser_df: pd.DataFrame) -> UiRunBrowserFilterOptions:
    # Опции фильтров собираем из merged-таблицы запуска, чтобы control-panel знал только про готовые списки.
    return UiRunBrowserFilterOptions(
        final_domain_states=_ordered_unique_labels(
            run_browser_df,
            column_name="final_domain_state",
            preferred_order=("id", "unknown", "ood"),
        ),
        priority_labels=_ordered_unique_labels(
            run_browser_df,
            column_name="priority_label",
            preferred_order=("high", "medium", "low"),
        ),
        spec_classes=_ordered_unique_labels(
            run_browser_df,
            column_name="spec_class",
            preferred_order=(),
        ),
    )


def apply_ui_run_browser_filters(
    run_browser_df: pd.DataFrame,
    filters: UiRunBrowserFilters,
) -> pd.DataFrame:
    # Применяем фильтры последовательно и без побочных эффектов, чтобы unit-тесты могли проверять каждый шаг.
    filtered_df = run_browser_df.copy()

    if filters.final_domain_states:
        filtered_df = filtered_df.loc[
            filtered_df["final_domain_state"].astype(str).isin(filters.final_domain_states),
            :,
        ]
    if filters.priority_labels:
        filtered_df = filtered_df.loc[
            filtered_df["priority_label"].astype(str).isin(filters.priority_labels),
            :,
        ]
    if filters.spec_classes:
        filtered_df = filtered_df.loc[
            filtered_df["spec_class"].astype(str).isin(filters.spec_classes),
            :,
        ]

    source_id_query = _normalize_query(filters.source_id_query)
    if source_id_query is not None:
        filtered_df = filtered_df.loc[
            filtered_df["source_id"].astype(str).str.contains(source_id_query, regex=False),
            :,
        ]

    return filtered_df.reset_index(drop=True)


def build_ui_filtered_domain_distribution_frame(filtered_df: pd.DataFrame) -> pd.DataFrame:
    # После фильтрации distribution должен считаться от текущей выборки, а не от всего run.
    return _build_distribution_frame(
        filtered_df,
        column_name="final_domain_state",
        label_name="final_domain_state",
        drop_missing=False,
    )


def build_ui_filtered_priority_distribution_frame(filtered_df: pd.DataFrame) -> pd.DataFrame:
    # Для priority-распределения пропускаем строки без ranking, чтобы unknown/OOD не превращались в `nan`.
    return _build_distribution_frame(
        filtered_df,
        column_name="priority_label",
        label_name="priority_label",
        drop_missing=True,
    )


def build_ui_run_browser_preview_frame(
    filtered_df: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    # Визуальный preview ограничиваем top_n, но export должен оставаться полным.
    return filtered_df.head(max(1, int(top_n))).copy()


def build_ui_run_browser_export_bytes(filtered_df: pd.DataFrame) -> bytes:
    # Для `download_button` заранее готовим CSV в памяти и отдаём готовый byte payload.
    return filtered_df.to_csv(index=False).encode("utf-8")


def _ordered_unique_labels(
    df: pd.DataFrame,
    *,
    column_name: str,
    preferred_order: tuple[str, ...],
) -> tuple[str, ...]:
    if column_name not in df.columns:
        return ()

    unique_values = {
        str(value)
        for value in df[column_name].dropna().astype(str).tolist()
        if str(value).strip()
    }
    if not unique_values:
        return ()

    ordered_values: list[str] = [
        label for label in preferred_order if label in unique_values
    ]
    ordered_values.extend(
        sorted(label for label in unique_values if label not in preferred_order)
    )
    return tuple(ordered_values)


def _normalize_query(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _build_distribution_frame(
    df: pd.DataFrame,
    *,
    column_name: str,
    label_name: str,
    drop_missing: bool,
) -> pd.DataFrame:
    if df.empty or column_name not in df.columns:
        return pd.DataFrame(columns=[label_name, "n_rows", "share"])

    values = df[column_name]
    if drop_missing:
        values = values.dropna()
        if values.empty:
            return pd.DataFrame(columns=[label_name, "n_rows", "share"])

    value_counts = (
        values.astype(str)
        .loc[lambda series: series.str.strip().ne("")]
        .value_counts(dropna=False)
        .rename_axis(label_name)
        .reset_index(name="n_rows")
    )
    if value_counts.empty:
        return pd.DataFrame(columns=[label_name, "n_rows", "share"])

    n_rows_total = int(value_counts["n_rows"].sum())
    value_counts["share"] = value_counts["n_rows"].astype("float64") / float(n_rows_total)
    return value_counts.copy()


__all__ = [
    "UiRunBrowserFilterOptions",
    "UiRunBrowserFilters",
    "apply_ui_run_browser_filters",
    "build_ui_filtered_domain_distribution_frame",
    "build_ui_filtered_priority_distribution_frame",
    "build_ui_run_browser_export_bytes",
    "build_ui_run_browser_filter_options",
    "build_ui_run_browser_preview_frame",
]
