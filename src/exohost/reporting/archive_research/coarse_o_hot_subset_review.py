# Review-хелперы для physically hot `O/B-like` subset внутри true `O` source.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from exohost.datasets.archive_research.load_coarse_o_review_dataset import (
    load_coarse_o_review_dataset,
)
from exohost.db.engine import make_read_only_engine
from exohost.reporting.archive_research.coarse_o_review import (
    build_coarse_o_final_outcome_frame,
    build_coarse_o_scored_frame,
    build_o_final_coarse_distribution_frame,
    build_o_final_outcome_distribution_frame,
    build_o_final_reason_frame,
    build_o_predicted_physics_summary_frame,
    build_o_scored_prediction_frame,
    build_o_source_quality_reason_frame,
    build_o_source_quality_summary_frame,
)


@dataclass(frozen=True, slots=True)
class HotOBLikeSubsetConfig:
    # Project baseline: B-type stars start at roughly 10 000 K, so this cut keeps
    # only physically hot `O/B-like` rows inside the true `O` source.
    teff_min_k: float = 10_000.0


@dataclass(frozen=True, slots=True)
class HotOBLikeSubsetReviewBundle:
    # Полный пакет review для physically hot `O/B-like` subset.
    config: HotOBLikeSubsetConfig
    source_df: pd.DataFrame
    hot_subset_source_df: pd.DataFrame
    pass_hot_subset_df: pd.DataFrame
    scored_pass_hot_subset_df: pd.DataFrame
    final_hot_subset_df: pd.DataFrame


_DEFAULT_HOT_OB_CONFIG = HotOBLikeSubsetConfig()


def load_hot_ob_like_subset_review_bundle_from_env(
    *,
    coarse_model_run_dir: str | Path,
    final_decision_run_dir: str | Path,
    config: HotOBLikeSubsetConfig = _DEFAULT_HOT_OB_CONFIG,
    source_limit: int | None = None,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> HotOBLikeSubsetReviewBundle:
    # Загружаем live true `O` source из БД и строим hot-subset review bundle.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        source_df = load_coarse_o_review_dataset(engine, limit=source_limit)
    finally:
        engine.dispose()
    return build_hot_ob_like_subset_review_bundle(
        source_df,
        coarse_model_run_dir=coarse_model_run_dir,
        final_decision_run_dir=final_decision_run_dir,
        config=config,
    )


def build_hot_ob_like_subset_review_bundle(
    source_df: pd.DataFrame,
    *,
    coarse_model_run_dir: str | Path,
    final_decision_run_dir: str | Path,
    config: HotOBLikeSubsetConfig = _DEFAULT_HOT_OB_CONFIG,
) -> HotOBLikeSubsetReviewBundle:
    # Строим review bundle только для physically hot true `O` rows.
    hot_subset_source_df = build_hot_ob_like_subset_source_frame(source_df, config=config)
    pass_hot_subset_df = _build_pass_hot_subset_frame(hot_subset_source_df)
    scored_pass_hot_subset_df = build_coarse_o_scored_frame(
        pass_hot_subset_df,
        coarse_model_run_dir=coarse_model_run_dir,
    )
    final_hot_subset_df = build_coarse_o_final_outcome_frame(
        hot_subset_source_df,
        final_decision_run_dir=final_decision_run_dir,
    )
    return HotOBLikeSubsetReviewBundle(
        config=config,
        source_df=source_df,
        hot_subset_source_df=hot_subset_source_df,
        pass_hot_subset_df=pass_hot_subset_df,
        scored_pass_hot_subset_df=scored_pass_hot_subset_df,
        final_hot_subset_df=final_hot_subset_df,
    )


def build_hot_ob_like_subset_source_frame(
    source_df: pd.DataFrame,
    *,
    config: HotOBLikeSubsetConfig = _DEFAULT_HOT_OB_CONFIG,
) -> pd.DataFrame:
    # Выделяем только physically hot `O/B-like` rows по baseline `teff_gspphot`.
    teff_series = pd.to_numeric(source_df["teff_gspphot"], errors="coerce")
    if not isinstance(teff_series, pd.Series):
        raise TypeError("Expected pandas Series after teff_gspphot normalization.")
    hot_mask = teff_series.ge(config.teff_min_k)
    return source_df.loc[hot_mask].copy().reset_index(drop=True)


def build_hot_subset_membership_summary_frame(
    bundle: HotOBLikeSubsetReviewBundle,
) -> pd.DataFrame:
    # Компактная сводка по объему hot-subset относительно полного `O` source.
    n_rows_source = int(bundle.source_df.shape[0])
    n_rows_hot_subset = int(bundle.hot_subset_source_df.shape[0])
    n_rows_hot_pass = int(bundle.pass_hot_subset_df.shape[0])
    n_rows_hot_scored = int(bundle.scored_pass_hot_subset_df.shape[0])
    return pd.DataFrame(
        [
            {
                "teff_min_k": float(bundle.config.teff_min_k),
                "n_rows_true_o_source": n_rows_source,
                "n_rows_hot_subset": n_rows_hot_subset,
                "share_hot_subset_in_source": (
                    float(n_rows_hot_subset / n_rows_source) if n_rows_source > 0 else 0.0
                ),
                "n_rows_hot_pass": n_rows_hot_pass,
                "share_hot_pass_in_hot_subset": (
                    float(n_rows_hot_pass / n_rows_hot_subset)
                    if n_rows_hot_subset > 0
                    else 0.0
                ),
                "n_rows_hot_scored": n_rows_hot_scored,
            }
        ]
    )


def build_hot_subset_quality_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Распределение quality-state внутри physically hot subset.
    return build_o_source_quality_summary_frame(df)


def build_hot_subset_quality_reason_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Top quality reasons внутри physically hot subset.
    return build_o_source_quality_reason_frame(df, top_n=top_n)


def build_hot_subset_scored_prediction_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Распределение coarse predictions на hot pass-subset.
    return build_o_scored_prediction_frame(df, top_n=top_n)


def build_hot_subset_predicted_physics_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Медианные физические параметры по predicted groups на hot pass-subset.
    return build_o_predicted_physics_summary_frame(df)


def build_hot_subset_final_outcome_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Final-domain outcome только для hot-subset.
    return build_o_final_outcome_distribution_frame(df)


def build_hot_subset_final_coarse_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Final coarse distribution только для hot-subset.
    return build_o_final_coarse_distribution_frame(df)


def build_hot_subset_final_reason_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Top final-decision reasons только для hot-subset.
    return build_o_final_reason_frame(df, top_n=top_n)


def build_hot_subset_high_confidence_non_ob_preview_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    # Самые уверенные hot-subset случаи, где coarse уводит объект не в `O/B`.
    preview_columns = [
        name
        for name in (
            "source_id",
            "coarse_predicted_label",
            "coarse_probability_max",
            "coarse_probability_margin",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_flame",
        )
        if name in df.columns
    ]
    predicted_label_series = df["coarse_predicted_label"].astype("string").str.upper()
    filtered_df = df.loc[
        ~predicted_label_series.isin({"O", "B"}),
        preview_columns,
    ].copy()
    return filtered_df.sort_values(
        ["coarse_probability_max", "coarse_probability_margin"],
        ascending=[False, False],
        kind="mergesort",
        ignore_index=True,
    ).head(top_n)


def _build_pass_hot_subset_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.loc[df["quality_state"].astype(str) == "pass"].copy().reset_index(drop=True)
    )
