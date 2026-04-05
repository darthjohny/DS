# Frame-builders для review feature separability `O vs B`.

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.reporting.archive_research.coarse_ob_feature_separability_contracts import (
    DEFAULT_COARSE_OB_SEPARABILITY_CONFIG,
    CoarseOBFeatureSeparabilityConfig,
    CoarseOBFeatureSeparabilityReviewBundle,
)


def build_train_time_ob_boundary_frame(
    source_df: pd.DataFrame,
    *,
    config: CoarseOBFeatureSeparabilityConfig = DEFAULT_COARSE_OB_SEPARABILITY_CONFIG,
) -> pd.DataFrame:
    # Выделяем train-time hot `O/B` boundary source без quality-gate слоя.
    spec_class_series = source_df["spec_class"].astype("string").str.upper()
    hot_mask = source_df["teff_gspphot"].map(
        lambda value: (_to_optional_float(value) or float("-inf")) >= config.hot_teff_min_k
    )
    boundary_mask = spec_class_series.isin({"O", "B"}) & hot_mask
    return source_df.loc[boundary_mask].copy().reset_index(drop=True)


def build_boundary_membership_summary_frame(
    bundle: CoarseOBFeatureSeparabilityReviewBundle,
) -> pd.DataFrame:
    # Короткая сводка по train-time `O/B` boundary source.
    return pd.DataFrame(
        [
            {
                "hot_teff_min_k": float(bundle.config.hot_teff_min_k),
                "n_rows_source": int(bundle.source_df.shape[0]),
                "n_rows_boundary": int(bundle.boundary_df.shape[0]),
                "n_rows_scored": int(bundle.scored_boundary_df.shape[0]),
            }
        ]
    )


def build_boundary_true_class_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Сколько true `O` и true `B` строк входит в train-time boundary source.
    return _build_distribution_frame(
        df,
        column_name="spec_class",
        label_name="true_spectral_class",
    )


def build_boundary_predicted_class_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Во что coarse-модель переводит train-time `O/B` boundary source.
    return _build_distribution_frame(
        df,
        column_name="coarse_predicted_label",
        label_name="coarse_predicted_label",
    )


def build_boundary_probability_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Сравниваем `P(O)` и `P(B)` на true `O` и true `B`.
    required_columns = {"spec_class", "coarse_probability__O", "coarse_probability__B"}
    if not required_columns.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "true_spectral_class",
                "n_rows",
                "median_probability__O",
                "median_probability__B",
                "mean_probability__O",
                "mean_probability__B",
            ]
        )
    grouped_df = (
        df.groupby("spec_class", dropna=False, sort=True)
        .agg(
            n_rows=("source_id", "size"),
            median_probability__O=("coarse_probability__O", "median"),
            median_probability__B=("coarse_probability__B", "median"),
            mean_probability__O=("coarse_probability__O", "mean"),
            mean_probability__B=("coarse_probability__B", "mean"),
        )
        .reset_index()
        .rename(columns={"spec_class": "true_spectral_class"})
    )
    return grouped_df.sort_values(
        ["true_spectral_class"],
        ascending=[True],
        kind="mergesort",
        ignore_index=True,
    )


def build_boundary_feature_physics_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Сравниваем медианную физику true `O` и true `B` на boundary source.
    if df.empty:
        return pd.DataFrame(
            columns=[
                "true_spectral_class",
                "n_rows",
                "median_teff_gspphot",
                "median_logg_gspphot",
                "median_mh_gspphot",
                "median_bp_rp",
                "median_parallax",
                "median_parallax_over_error",
                "median_ruwe",
                "median_radius_feature",
            ]
        )
    grouped_df = (
        df.groupby("spec_class", dropna=False, sort=True)
        .agg(
            n_rows=("source_id", "size"),
            median_teff_gspphot=("teff_gspphot", "median"),
            median_logg_gspphot=("logg_gspphot", "median"),
            median_mh_gspphot=("mh_gspphot", "median"),
            median_bp_rp=("bp_rp", "median"),
            median_parallax=("parallax", "median"),
            median_parallax_over_error=("parallax_over_error", "median"),
            median_ruwe=("ruwe", "median"),
            median_radius_feature=("radius_feature", "median"),
        )
        .reset_index()
        .rename(columns={"spec_class": "true_spectral_class"})
    )
    return grouped_df.sort_values(
        ["true_spectral_class"],
        ascending=[True],
        kind="mergesort",
        ignore_index=True,
    )


def build_univariate_separability_auc_frame(
    df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = GAIA_ID_COARSE_CLASSIFICATION_TASK.feature_columns,
) -> pd.DataFrame:
    # Считаем single-feature separability для `O vs B` через ROC AUC.
    if df.empty:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "n_rows_used",
                "auc_ovr_o",
                "separability_auc",
                "higher_value_class",
                "median_o",
                "median_b",
                "median_gap",
            ]
        )

    rows: list[dict[str, object]] = []
    target_series = df["spec_class"].astype("string").str.upper()

    for feature_name in feature_columns:
        if feature_name not in df.columns:
            continue
        numeric_series = df[feature_name].map(_to_optional_float)
        valid_mask = numeric_series.notna() & target_series.isin({"O", "B"})
        if not bool(valid_mask.any()):
            continue

        valid_df = pd.DataFrame(
            {
                "spec_class": target_series.loc[valid_mask].astype("string"),
                "feature_value": numeric_series.loc[valid_mask].astype("float64"),
            }
        )
        if valid_df["spec_class"].nunique(dropna=True) < 2:
            continue

        target_binary = valid_df["spec_class"].map(lambda value: 1 if value == "O" else 0)
        auc_ovr_o = float(roc_auc_score(target_binary, valid_df["feature_value"]))
        separability_auc = max(auc_ovr_o, 1.0 - auc_ovr_o)
        median_o = _series_median(
            valid_df.loc[valid_df["spec_class"] == "O", "feature_value"]
        )
        median_b = _series_median(
            valid_df.loc[valid_df["spec_class"] == "B", "feature_value"]
        )
        rows.append(
            {
                "feature_name": feature_name,
                "n_rows_used": int(valid_df.shape[0]),
                "auc_ovr_o": auc_ovr_o,
                "separability_auc": separability_auc,
                "higher_value_class": "O" if auc_ovr_o >= 0.5 else "B",
                "median_o": median_o,
                "median_b": median_b,
                "median_gap": float(median_o - median_b),
            }
        )

    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["separability_auc", "feature_name"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )


def _build_distribution_frame(
    df: pd.DataFrame,
    *,
    column_name: str,
    label_name: str,
) -> pd.DataFrame:
    if column_name not in df.columns:
        return pd.DataFrame(columns=[label_name, "n_rows", "share"])
    value_counts = df[column_name].astype("string").fillna("<NA>").value_counts(dropna=False)
    total_rows = int(value_counts.sum())
    rows: list[dict[str, object]] = []
    for label_value, n_rows in value_counts.items():
        rows.append(
            {
                label_name: str(label_value),
                "n_rows": int(n_rows),
                "share": float(n_rows / total_rows) if total_rows > 0 else 0.0,
            }
        )
    return pd.DataFrame.from_records(rows)


def _series_median(series: pd.Series) -> float:
    median_value = series.median()
    return float(median_value)


def _to_optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return None if pd.isna(value) else float(value)
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return float(stripped_value)
        except ValueError:
            return None
    return None
