# Файл `candidate_card.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - подготовку карточки одной звезды из готового run bundle;
# - сборку compact summary и блока физических параметров по `source_id`.
#
# Следующий слой:
# - компонент страницы карточки объекта;
# - unit-тесты helper-модуля UI.

from __future__ import annotations

import pandas as pd

from exohost.ui.loaders import UiLoadedRunBundle

SUMMARY_COLUMNS: tuple[str, ...] = (
    "source_id",
    "final_domain_state",
    "final_quality_state",
    "final_coarse_class",
    "final_refinement_label",
    "final_refinement_state",
    "final_decision_reason",
    "quality_reason",
    "review_bucket",
    "priority_label",
    "priority_score",
    "priority_reason",
    "host_similarity_score",
    "observability_score",
)

PHYSICS_COLUMNS: tuple[str, ...] = (
    "source_id",
    "spec_class",
    "spec_subclass",
    "evolution_stage",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "phot_g_mean_mag",
    "radius_flame",
    "lum_flame",
    "evolstage_flame",
)


def build_ui_candidate_source_options(bundle: UiLoadedRunBundle) -> tuple[str, ...]:
    # В выпадающем списке держим только реальные source_id из готового final-decision bundle.
    merged_df = _build_candidate_lookup_frame(bundle)
    if merged_df.empty:
        return ()

    source_ids = merged_df.loc[:, "source_id"].drop_duplicates().tolist()
    return tuple(str(source_id) for source_id in source_ids)


def build_ui_candidate_summary_frame(
    bundle: UiLoadedRunBundle,
    source_id: str | int,
) -> pd.DataFrame:
    # Summary-блок показывает маршрут по pipeline и причины итогового решения.
    candidate_row = _extract_candidate_row(bundle, source_id)
    if candidate_row.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return candidate_row.loc[:, list(SUMMARY_COLUMNS)].copy()


def build_ui_candidate_physics_frame(
    bundle: UiLoadedRunBundle,
    source_id: str | int,
) -> pd.DataFrame:
    # Physical preview держим отдельно, чтобы карточка не смешивала маршрут и астрофизику.
    candidate_row = _extract_candidate_row(bundle, source_id)
    if candidate_row.empty:
        return pd.DataFrame(columns=PHYSICS_COLUMNS)
    return candidate_row.loc[:, list(PHYSICS_COLUMNS)].copy()


def _extract_candidate_row(
    bundle: UiLoadedRunBundle,
    source_id: str | int,
) -> pd.DataFrame:
    lookup_df = _build_candidate_lookup_frame(bundle)
    if lookup_df.empty:
        return pd.DataFrame(columns=tuple(dict.fromkeys((*SUMMARY_COLUMNS, *PHYSICS_COLUMNS))))

    source_id_key = str(source_id)
    filtered_df = lookup_df.loc[
        lookup_df["source_id"].astype(str) == source_id_key,
        :,
    ].copy()
    if filtered_df.empty:
        return pd.DataFrame(columns=tuple(dict.fromkeys((*SUMMARY_COLUMNS, *PHYSICS_COLUMNS))))

    for column_name in (*SUMMARY_COLUMNS, *PHYSICS_COLUMNS):
        if column_name not in filtered_df.columns:
            filtered_df[column_name] = pd.NA
    return filtered_df.reset_index(drop=True)


def _build_candidate_lookup_frame(bundle: UiLoadedRunBundle) -> pd.DataFrame:
    # Собираем единый one-row-per-source lookup, чтобы страница не делала несколько merge по месту.
    merged_df = bundle.loaded_artifacts.final_decision_df.copy()

    for extra_df in (
        bundle.loaded_artifacts.decision_input_df,
        bundle.loaded_artifacts.priority_input_df,
        bundle.loaded_artifacts.priority_ranking_df,
    ):
        merged_df = _merge_new_columns(
            merged_df,
            extra_df,
        )

    if "source_id" in merged_df.columns:
        merged_df = merged_df.sort_values(
            "source_id",
            ascending=True,
            kind="mergesort",
            ignore_index=True,
        )
    return merged_df


def _merge_new_columns(base_df: pd.DataFrame, extra_df: pd.DataFrame) -> pd.DataFrame:
    if extra_df.empty or "source_id" not in extra_df.columns:
        return base_df

    columns_to_add = [
        column_name
        for column_name in extra_df.columns
        if column_name == "source_id" or column_name not in base_df.columns
    ]
    return base_df.merge(
        extra_df.loc[:, columns_to_add],
        on="source_id",
        how="left",
        validate="one_to_one",
    )


__all__ = [
    "PHYSICS_COLUMNS",
    "SUMMARY_COLUMNS",
    "build_ui_candidate_physics_frame",
    "build_ui_candidate_source_options",
    "build_ui_candidate_summary_frame",
]
