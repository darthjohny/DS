# Файл `host_calibration_review.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import pandas as pd

from exohost.reporting.binary_calibration_review import (
    DEFAULT_BINARY_CALIBRATION_CONFIG,
    BinaryCalibrationConfig,
    build_binary_calibration_curve_frame,
    build_binary_calibration_summary_frame,
    build_binary_probability_bin_frame,
)
from exohost.reporting.host_calibration_source import HostCalibrationSource


def build_host_calibration_split_summary_frame(
    source: HostCalibrationSource,
) -> pd.DataFrame:
    # One-row summary по reconstructed host calibration source.
    train_target = _require_series_column(source.train_scored_df, source.target_column)
    test_target = _require_series_column(source.test_scored_df, source.target_column)
    return pd.DataFrame(
        [
            {
                "task_name": source.task_name,
                "model_name": source.model_name,
                "n_rows_full": int(source.split.full_df.shape[0]),
                "n_rows_train": int(source.split.train_df.shape[0]),
                "n_rows_test": int(source.split.test_df.shape[0]),
                "train_positive_rate": float(
                    train_target.astype(str).eq(source.positive_label).mean()
                ),
                "test_positive_rate": float(
                    test_target.astype(str).eq(source.positive_label).mean()
                ),
            }
        ]
    )


def build_host_calibration_metric_summary_frame(
    source: HostCalibrationSource,
    *,
    config: BinaryCalibrationConfig = DEFAULT_BINARY_CALIBRATION_CONFIG,
) -> pd.DataFrame:
    # Core calibration metrics по host holdout predictions.
    test_target = _require_series_column(source.test_scored_df, source.target_column)
    test_score = _require_series_column(source.test_scored_df, source.host_score_column)
    return build_binary_calibration_summary_frame(
        test_target,
        test_score,
        config=BinaryCalibrationConfig(
            positive_label=source.positive_label,
            n_bins=config.n_bins,
            strategy=config.strategy,
        ),
    )


def build_host_calibration_curve_review_frame(
    source: HostCalibrationSource,
    *,
    config: BinaryCalibrationConfig = DEFAULT_BINARY_CALIBRATION_CONFIG,
) -> pd.DataFrame:
    # Reliability curve по holdout host predictions.
    test_target = _require_series_column(source.test_scored_df, source.target_column)
    test_score = _require_series_column(source.test_scored_df, source.host_score_column)
    return build_binary_calibration_curve_frame(
        test_target,
        test_score,
        config=BinaryCalibrationConfig(
            positive_label=source.positive_label,
            n_bins=config.n_bins,
            strategy=config.strategy,
        ),
    )


def build_host_probability_bin_review_frame(
    source: HostCalibrationSource,
    *,
    config: BinaryCalibrationConfig = DEFAULT_BINARY_CALIBRATION_CONFIG,
) -> pd.DataFrame:
    # Bin-level breakdown по holdout host probabilities.
    test_target = _require_series_column(source.test_scored_df, source.target_column)
    test_score = _require_series_column(source.test_scored_df, source.host_score_column)
    return build_binary_probability_bin_frame(
        test_target,
        test_score,
        config=BinaryCalibrationConfig(
            positive_label=source.positive_label,
            n_bins=config.n_bins,
            strategy=config.strategy,
        ),
    )


def build_host_calibration_group_frame(
    source: HostCalibrationSource,
    *,
    group_column: str,
) -> pd.DataFrame:
    # Смотрим calibration-context по `spec_class` или `evolution_stage` на holdout.
    if group_column not in source.test_scored_df.columns:
        return pd.DataFrame(
            columns=[
                group_column,
                "n_rows",
                "positive_rate",
                "mean_host_similarity_score",
                "median_host_similarity_score",
            ]
        )

    review_df = source.test_scored_df.loc[
        :,
        [group_column, source.target_column, source.host_score_column],
    ].copy()
    review_df[source.host_score_column] = pd.to_numeric(
        review_df[source.host_score_column],
        errors="coerce",
    )
    review_df = review_df.dropna(subset=[source.host_score_column])
    if review_df.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                "n_rows",
                "positive_rate",
                "mean_host_similarity_score",
                "median_host_similarity_score",
            ]
        )

    review_df["target_positive"] = (
        review_df[source.target_column].astype(str).eq(source.positive_label).astype(int)
    )
    grouped_df = (
        review_df.groupby(group_column, dropna=False, sort=True)
        .agg(
            n_rows=("target_positive", "size"),
            positive_rate=("target_positive", "mean"),
            mean_host_similarity_score=(source.host_score_column, "mean"),
            median_host_similarity_score=(source.host_score_column, "median"),
        )
        .reset_index()
        .sort_values("n_rows", ascending=False, kind="mergesort", ignore_index=True)
    )
    grouped_df[group_column] = grouped_df[group_column].astype(str)
    return grouped_df


def _require_series_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column
