# Тестовый файл `test_binary_calibration_review.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, cast

import pandas as pd
import pytest

from exohost.reporting.binary_calibration_review import (
    BinaryCalibrationConfig,
    build_binary_calibration_curve_frame,
    build_binary_calibration_summary_frame,
    build_binary_probability_bin_frame,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _require_float_scalar(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}.")
    return float(value)


def test_binary_calibration_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        BinaryCalibrationConfig(n_bins=1)

    with pytest.raises(ValueError, match="quantile' or 'uniform"):
        BinaryCalibrationConfig(strategy=cast(Any, "median"))


def test_binary_calibration_summary_frame_returns_core_metrics() -> None:
    y_true = pd.Series(["host", "field", "host", "field", "host"], dtype="string")
    y_score = pd.Series([0.91, 0.10, 0.78, 0.32, 0.64], dtype="float64")

    summary_df = build_binary_calibration_summary_frame(y_true, y_score)

    assert _require_int_scalar(summary_df.loc[0, "n_rows"]) == 5
    assert _require_float_scalar(summary_df.loc[0, "positive_rate"]) == pytest.approx(0.6)
    assert _require_float_scalar(summary_df.loc[0, "mean_predicted_probability"]) == pytest.approx(
        0.55
    )
    assert _require_float_scalar(summary_df.loc[0, "brier_score"]) >= 0.0
    assert _require_float_scalar(summary_df.loc[0, "log_loss"]) >= 0.0
    assert _require_float_scalar(summary_df.loc[0, "roc_auc"]) >= 0.0


def test_binary_calibration_curve_frame_builds_expected_columns() -> None:
    y_true = pd.Series(["host", "field", "host", "field", "host", "field"], dtype="string")
    y_score = pd.Series([0.95, 0.05, 0.80, 0.25, 0.70, 0.30], dtype="float64")

    curve_df = build_binary_calibration_curve_frame(
        y_true,
        y_score,
        config=BinaryCalibrationConfig(n_bins=3, strategy="uniform"),
    )

    assert list(curve_df.columns) == [
        "bin_index",
        "mean_predicted_probability",
        "fraction_of_positives",
    ]
    assert not curve_df.empty


def test_binary_probability_bin_frame_returns_bin_level_summary() -> None:
    y_true = pd.Series(["host", "field", "host", "field", "host", "field"], dtype="string")
    y_score = pd.Series([0.95, 0.05, 0.80, 0.25, 0.70, 0.30], dtype="float64")

    review_df = build_binary_probability_bin_frame(
        y_true,
        y_score,
        config=BinaryCalibrationConfig(n_bins=5),
    )

    assert "probability_bin" in review_df.columns
    assert "positive_rate" in review_df.columns
    assert _require_int_scalar(review_df["n_rows"].sum()) == 6
