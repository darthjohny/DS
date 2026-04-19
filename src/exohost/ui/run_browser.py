# Файл `run_browser.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - подготовку read-only таблиц для страницы просмотра готового запуска;
# - компактные сводки по domain-state, priority и merged-таблицу запуска.
#
# Следующий слой:
# - компонент страницы просмотра запуска;
# - unit-тесты helper-модуля UI.

from __future__ import annotations

from numbers import Integral

import pandas as pd

from exohost.ui.loaders import UiLoadedRunBundle

TOP_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "source_id",
    "final_domain_state",
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


def build_ui_domain_distribution_frame(bundle: UiLoadedRunBundle) -> pd.DataFrame:
    # Domain-state распределение читаем прямо из final_decision, потому что это итоговый routing.
    distribution_df = _build_distribution_frame(
        bundle.loaded_artifacts.final_decision_df,
        column_name="final_domain_state",
        label_name="final_domain_state",
    )
    return distribution_df.copy()


def build_ui_priority_distribution_frame(bundle: UiLoadedRunBundle) -> pd.DataFrame:
    # Priority распределение берем из ranking output, чтобы показывать уже готовый shortlist.
    distribution_df = _build_distribution_frame(
        bundle.loaded_artifacts.priority_ranking_df,
        column_name="priority_label",
        label_name="priority_label",
    )
    return distribution_df.copy()


def build_ui_top_candidates_frame(
    bundle: UiLoadedRunBundle,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    # Верхний shortlist собираем из полной merged-таблицы запуска, но оставляем только ranked rows.
    run_browser_df = build_ui_run_browser_frame(bundle)
    if run_browser_df.empty:
        return pd.DataFrame(columns=TOP_CANDIDATE_COLUMNS)

    ranked_df = run_browser_df.loc[
        run_browser_df["priority_label"].notna(),
        :,
    ].reset_index(drop=True)
    return ranked_df.loc[:, list(TOP_CANDIDATE_COLUMNS)].head(top_n).copy()


def build_ui_run_browser_frame(bundle: UiLoadedRunBundle) -> pd.DataFrame:
    # Готовим единый read-only frame по запуску, чтобы page-слой не делал merge и сортировку по месту.
    final_decision_df = bundle.loaded_artifacts.final_decision_df
    if final_decision_df.empty:
        return pd.DataFrame(columns=TOP_CANDIDATE_COLUMNS)

    merged_df = final_decision_df.copy()
    merged_df = _merge_selected_new_columns(
        merged_df,
        bundle.loaded_artifacts.priority_ranking_df,
        selected_columns=(
            "priority_label",
            "priority_score",
            "priority_reason",
            "host_similarity_score",
            "observability_score",
        ),
    )
    merged_df = _merge_selected_new_columns(
        merged_df,
        bundle.loaded_artifacts.priority_input_df,
        selected_columns=(
            "host_similarity_score",
            "observability_score",
        ),
    )
    merged_df = _merge_selected_new_columns(
        merged_df,
        bundle.loaded_artifacts.decision_input_df,
        selected_columns=(
            "spec_class",
            "spec_subclass",
            "quality_reason",
            "review_bucket",
        ),
    )

    sort_ready = merged_df.copy()
    for column_name in (
        "priority_score",
        "host_similarity_score",
        "observability_score",
    ):
        if column_name in sort_ready.columns:
            sort_ready[column_name] = pd.to_numeric(
                sort_ready[column_name],
                errors="coerce",
            )

    sort_ready = sort_ready.sort_values(
        [
            "priority_score",
            "host_similarity_score",
            "observability_score",
            "source_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
        ignore_index=True,
        na_position="last",
    )

    for column_name in TOP_CANDIDATE_COLUMNS:
        if column_name not in sort_ready.columns:
            sort_ready[column_name] = pd.NA
    return sort_ready.copy()


def _merge_selected_new_columns(
    base_df: pd.DataFrame,
    extra_df: pd.DataFrame,
    *,
    selected_columns: tuple[str, ...],
) -> pd.DataFrame:
    if extra_df.empty or "source_id" not in extra_df.columns:
        return base_df

    columns_to_add = [
        column_name
        for column_name in selected_columns
        if column_name in extra_df.columns and column_name not in base_df.columns
    ]
    if not columns_to_add:
        return base_df

    return base_df.merge(
        extra_df.loc[:, ["source_id", *columns_to_add]],
        on="source_id",
        how="left",
        validate="one_to_one",
    )


def _build_distribution_frame(
    df: pd.DataFrame,
    *,
    column_name: str,
    label_name: str,
) -> pd.DataFrame:
    if df.empty or column_name not in df.columns:
        return pd.DataFrame(columns=[label_name, "n_rows", "share"])

    value_counts = (
        df[column_name]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(label_name)
        .reset_index(name="n_rows")
    )
    n_rows_total = _require_int_scalar(value_counts["n_rows"].sum())
    if n_rows_total == 0:
        value_counts["share"] = pd.Series(dtype="float64")
        return value_counts

    value_counts["share"] = value_counts["n_rows"].astype("float64") / float(n_rows_total)
    return value_counts.copy()


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral):
        return int(value)
    raise TypeError(f"Expected integer-like scalar, got {type(value)!r}.")


__all__ = [
    "TOP_CANDIDATE_COLUMNS",
    "build_ui_domain_distribution_frame",
    "build_ui_priority_distribution_frame",
    "build_ui_run_browser_frame",
    "build_ui_top_candidates_frame",
]
