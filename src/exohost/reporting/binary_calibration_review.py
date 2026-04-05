# Файл `binary_calibration_review.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

_DEFAULT_N_BINS = 10
CalibrationStrategy = Literal["quantile", "uniform"]


@dataclass(frozen=True, slots=True)
class BinaryCalibrationConfig:
    # Конфиг lightweight reliability-review без вмешательства в обучение модели.
    positive_label: str = "host"
    n_bins: int = _DEFAULT_N_BINS
    strategy: CalibrationStrategy = "quantile"

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError("BinaryCalibrationConfig.n_bins must be at least 2.")
        if self.strategy not in {"quantile", "uniform"}:
            raise ValueError(
                "BinaryCalibrationConfig.strategy must be 'quantile' or 'uniform'."
            )


DEFAULT_BINARY_CALIBRATION_CONFIG = BinaryCalibrationConfig()


def build_binary_calibration_summary_frame(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    config: BinaryCalibrationConfig = DEFAULT_BINARY_CALIBRATION_CONFIG,
) -> pd.DataFrame:
    # Возвращаем компактную quality-сводку для бинарного вероятностного сигнала.
    encoded_true = _encode_binary_targets(
        y_true,
        positive_label=config.positive_label,
    )
    score_series = _coerce_probability_series(y_score)
    aligned_true, aligned_score = _align_binary_inputs(encoded_true, score_series)
    if aligned_true.empty:
        return pd.DataFrame(
            [
                {
                    "n_rows": 0,
                    "positive_rate": 0.0,
                    "mean_predicted_probability": 0.0,
                    "brier_score": pd.NA,
                    "log_loss": pd.NA,
                    "roc_auc": pd.NA,
                }
            ]
        )

    positive_rate = float(aligned_true.mean())
    mean_probability = float(aligned_score.mean())
    brier = float(brier_score_loss(aligned_true, aligned_score))
    loss = float(log_loss(aligned_true, aligned_score, labels=[0, 1]))
    roc_auc = _safe_binary_roc_auc(aligned_true, aligned_score)

    return pd.DataFrame(
        [
            {
                "n_rows": int(aligned_true.shape[0]),
                "positive_rate": positive_rate,
                "mean_predicted_probability": mean_probability,
                "brier_score": brier,
                "log_loss": loss,
                "roc_auc": roc_auc,
            }
        ]
    )


def build_binary_calibration_curve_frame(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    config: BinaryCalibrationConfig = DEFAULT_BINARY_CALIBRATION_CONFIG,
) -> pd.DataFrame:
    # Строим reliability curve как tabular artifact для notebook-review.
    encoded_true = _encode_binary_targets(
        y_true,
        positive_label=config.positive_label,
    )
    score_series = _coerce_probability_series(y_score)
    aligned_true, aligned_score = _align_binary_inputs(encoded_true, score_series)
    if aligned_true.empty:
        return pd.DataFrame(
            columns=["bin_index", "mean_predicted_probability", "fraction_of_positives"]
        )

    fraction_of_positives, mean_predicted_probability = calibration_curve(
        aligned_true,
        aligned_score,
        n_bins=config.n_bins,
        strategy=config.strategy,
    )
    return pd.DataFrame(
        {
            "bin_index": list(range(1, len(fraction_of_positives) + 1)),
            "mean_predicted_probability": mean_predicted_probability.astype(float),
            "fraction_of_positives": fraction_of_positives.astype(float),
        }
    )


def build_binary_probability_bin_frame(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    config: BinaryCalibrationConfig = DEFAULT_BINARY_CALIBRATION_CONFIG,
) -> pd.DataFrame:
    # Показываем заполненность probability bins и долю positives в каждом бине.
    encoded_true = _encode_binary_targets(
        y_true,
        positive_label=config.positive_label,
    )
    score_series = _coerce_probability_series(y_score)
    aligned_true, aligned_score = _align_binary_inputs(encoded_true, score_series)
    if aligned_true.empty:
        return pd.DataFrame(
            columns=["probability_bin", "n_rows", "share", "positive_rate", "mean_probability"]
        )

    bin_edges = pd.interval_range(start=0.0, end=1.0, periods=config.n_bins)
    binned_score = pd.cut(
        aligned_score,
        bins=bin_edges,
        include_lowest=True,
    )
    probability_bin = pd.Series(
        [str(value) for value in binned_score],
        index=aligned_score.index,
        dtype="string",
    )
    review_df = pd.DataFrame(
        {
            "target_positive": aligned_true.astype(int),
            "predicted_probability": aligned_score.astype(float),
            "probability_bin": probability_bin,
        }
    )
    total_rows = int(review_df.shape[0])
    grouped = (
        review_df.groupby("probability_bin", dropna=False, sort=True)
        .agg(
            n_rows=("target_positive", "size"),
            positive_rate=("target_positive", "mean"),
            mean_probability=("predicted_probability", "mean"),
        )
        .reset_index()
    )
    grouped["share"] = grouped["n_rows"].astype(float) / float(total_rows)
    return grouped.loc[
        :,
        [
            "probability_bin",
            "n_rows",
            "share",
            "positive_rate",
            "mean_probability",
        ],
    ].copy()


def _encode_binary_targets(
    y_true: pd.Series,
    *,
    positive_label: str,
) -> pd.Series:
    target_series = y_true.astype("string")
    return target_series.eq(positive_label).astype("int64")


def _coerce_probability_series(y_score: pd.Series) -> pd.Series:
    score_series = pd.to_numeric(y_score, errors="coerce")
    if not isinstance(score_series, pd.Series):
        raise TypeError("Probability input must resolve to a pandas Series.")
    return cast(pd.Series, score_series)


def _align_binary_inputs(
    y_true: pd.Series,
    y_score: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    aligned_df = pd.DataFrame(
        {
            "target_positive": y_true.astype("int64"),
            "predicted_probability": y_score.astype("float64"),
        }
    ).dropna(subset=["predicted_probability"])
    return (
        aligned_df.loc[:, "target_positive"].astype("int64"),
        aligned_df.loc[:, "predicted_probability"].astype("float64"),
    )


def _safe_binary_roc_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    if y_true.nunique(dropna=False) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))
