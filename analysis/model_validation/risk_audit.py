"""Per-model stability profile и risk audit для heavy validation слоя."""

from __future__ import annotations

import pandas as pd

from analysis.model_validation.contracts import ModelValidationProtocol
from analysis.model_validation.scalars import scalar_to_float

LOW_MAX_RISK_SCORE = 1
MODERATE_MAX_RISK_SCORE = 4
AVG_GAP_WARNING_THRESHOLD = 0.05
AVG_GAP_FAIL_THRESHOLD = 0.10
MAX_GAP_WARNING_THRESHOLD = 0.08
MAX_GAP_FAIL_THRESHOLD = 0.15
TEST_STD_WARNING_THRESHOLD = 0.03
TEST_STD_FAIL_THRESHOLD = 0.06
TEST_RANGE_WARNING_THRESHOLD = 0.08
TEST_RANGE_FAIL_THRESHOLD = 0.15
CV_STD_WARNING_THRESHOLD = 0.04
CV_STD_FAIL_THRESHOLD = 0.08
BrierCalibrationStatus = str
def score_threshold_risk(
    value: float,
    *,
    warning_threshold: float,
    fail_threshold: float,
) -> int:
    """Преобразовать диагностическое значение в risk points."""
    if pd.isna(value):
        return 0
    if value > fail_threshold:
        return 2
    if value > warning_threshold:
        return 1
    return 0


def calibration_status_from_brier(test_brier: float) -> BrierCalibrationStatus:
    """Собрать coarse calibration verdict по averaged test Brier score."""
    if test_brier <= 0.10:
        return "good"
    if test_brier <= 0.20:
        return "watch"
    return "weak"


def risk_level_from_score(risk_score: int) -> str:
    """Преобразовать integer risk score в человекочитаемый verdict."""
    if risk_score <= LOW_MAX_RISK_SCORE:
        return "LOW"
    if risk_score <= MODERATE_MAX_RISK_SCORE:
        return "MODERATE"
    return "HIGH"


def select_audit_metric_rows(
    protocol: ModelValidationProtocol,
    generalization_summary_df: pd.DataFrame,
    *,
    model_name: str,
) -> tuple[str, pd.Series, pd.Series, pd.Series | None]:
    """Выбрать audit metric и train/test/cv строки для одной модели."""
    preferred_metric = protocol.comparison_protocol.search.refit_metric
    model_summary = generalization_summary_df[
        generalization_summary_df["model_name"] == model_name
    ]
    if model_summary.empty:
        raise ValueError(f"No generalization summary rows for model {model_name}.")

    metric_name = preferred_metric
    candidate_rows = model_summary[model_summary["metric_name"] == metric_name]
    if candidate_rows.empty:
        metric_name = "roc_auc"
        candidate_rows = model_summary[model_summary["metric_name"] == metric_name]
    if candidate_rows.empty:
        raise ValueError(
            "Heavy risk audit requires either protocol refit metric or roc_auc "
            f"rows for model {model_name}."
        )

    train_rows = candidate_rows[candidate_rows["stage_name"] == "train_in_sample"]
    test_rows = candidate_rows[candidate_rows["stage_name"] == "test_holdout"]
    cv_rows = candidate_rows[candidate_rows["stage_name"] == "cv_oof"]
    if train_rows.empty or test_rows.empty:
        raise ValueError(
            "Heavy risk audit requires both train_in_sample and test_holdout "
            f"rows for model {model_name} metric {metric_name}."
        )

    cv_row = cv_rows.iloc[0] if not cv_rows.empty else None
    return metric_name, train_rows.iloc[0], test_rows.iloc[0], cv_row


def collect_secondary_gap_risk(
    gap_diagnostics_df: pd.DataFrame,
    *,
    model_name: str,
    primary_metric: str,
) -> tuple[int, list[str]]:
    """Собрать дополнительные risk points по secondary метрикам.

    Основной audit metric нужен для стабильного baseline verdict, но для
    реального риска переобучения этого мало: модель может выглядеть
    приемлемо по `roc_auc`, но иметь явный разрыв по `pr_auc` или
    `precision_at_k`. Поэтому secondary train/test gaps учитываются
    отдельно и добавляют риск-баллы поверх primary verdict.
    """
    secondary_rows = gap_diagnostics_df[
        (gap_diagnostics_df["model_name"] == model_name)
        & (gap_diagnostics_df["metric_name"] != primary_metric)
    ].sort_values("metric_name", ignore_index=True)

    risk_score = 0
    reasons: list[str] = []
    for _, row in secondary_rows.iterrows():
        metric_name = str(row["metric_name"])
        avg_gap = scalar_to_float(row["abs_train_test_gap_mean"])
        max_gap = scalar_to_float(row["abs_train_test_gap_max"])

        avg_gap_risk = score_threshold_risk(
            avg_gap,
            warning_threshold=AVG_GAP_WARNING_THRESHOLD,
            fail_threshold=AVG_GAP_FAIL_THRESHOLD,
        )
        if avg_gap_risk > 0:
            risk_score += avg_gap_risk
            reasons.append(
                f"avg train/test gap for {metric_name} = {avg_gap:.3f}"
            )

        max_gap_risk = score_threshold_risk(
            max_gap,
            warning_threshold=MAX_GAP_WARNING_THRESHOLD,
            fail_threshold=MAX_GAP_FAIL_THRESHOLD,
        )
        if max_gap_risk > 0:
            risk_score += max_gap_risk
            reasons.append(
                f"max train/test gap for {metric_name} = {max_gap:.3f}"
            )

    return risk_score, reasons


def build_model_risk_audit_frame(
    protocol: ModelValidationProtocol,
    *,
    generalization_summary_df: pd.DataFrame,
    gap_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Собрать per-model stability profile и итоговый risk audit."""
    if generalization_summary_df.empty or gap_diagnostics_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for model_name in sorted(generalization_summary_df["model_name"].astype(str).unique()):
        metric_name, train_row, test_row, cv_row = select_audit_metric_rows(
            protocol,
            generalization_summary_df,
            model_name=model_name,
        )
        gap_rows = gap_diagnostics_df[
            (gap_diagnostics_df["model_name"] == model_name)
            & (gap_diagnostics_df["metric_name"] == metric_name)
        ]
        if gap_rows.empty:
            raise ValueError(
                "Heavy risk audit requires gap diagnostics row for "
                f"{model_name} / {metric_name}."
            )
        gap_row = gap_rows.iloc[0]

        brier_rows = generalization_summary_df[
            (generalization_summary_df["model_name"] == model_name)
            & (generalization_summary_df["metric_name"] == "brier")
            & (generalization_summary_df["stage_name"] == "test_holdout")
        ]
        test_brier_mean = (
            scalar_to_float(brier_rows.iloc[0]["score_mean"])
            if not brier_rows.empty
            else float("nan")
        )
        calibration_status = (
            calibration_status_from_brier(test_brier_mean)
            if not pd.isna(test_brier_mean)
            else "unknown"
        )

        avg_gap = abs(scalar_to_float(gap_row["train_minus_test_mean"]))
        max_gap = scalar_to_float(gap_row["abs_train_test_gap_max"])
        cv_gap_mean = abs(scalar_to_float(gap_row["cv_minus_test_mean"]))
        test_std = scalar_to_float(test_row["score_std"])
        test_range = scalar_to_float(test_row["score_max"]) - scalar_to_float(
            test_row["score_min"]
        )
        cv_std = (
            scalar_to_float(cv_row["score_std"]) if cv_row is not None else float("nan")
        )

        risk_score = 0
        risk_reasons: list[str] = []

        avg_gap_risk = score_threshold_risk(
            avg_gap,
            warning_threshold=AVG_GAP_WARNING_THRESHOLD,
            fail_threshold=AVG_GAP_FAIL_THRESHOLD,
        )
        if avg_gap_risk > 0:
            risk_score += avg_gap_risk
            risk_reasons.append(f"avg train/test gap for {metric_name} = {avg_gap:.3f}")

        max_gap_risk = score_threshold_risk(
            max_gap,
            warning_threshold=MAX_GAP_WARNING_THRESHOLD,
            fail_threshold=MAX_GAP_FAIL_THRESHOLD,
        )
        if max_gap_risk > 0:
            risk_score += max_gap_risk
            risk_reasons.append(f"max train/test gap for {metric_name} = {max_gap:.3f}")

        cv_gap_risk = score_threshold_risk(
            cv_gap_mean,
            warning_threshold=AVG_GAP_WARNING_THRESHOLD,
            fail_threshold=AVG_GAP_FAIL_THRESHOLD,
        )
        if cv_gap_risk > 0:
            risk_score += cv_gap_risk
            risk_reasons.append(f"avg cv/test gap for {metric_name} = {cv_gap_mean:.3f}")

        test_std_risk = score_threshold_risk(
            test_std,
            warning_threshold=TEST_STD_WARNING_THRESHOLD,
            fail_threshold=TEST_STD_FAIL_THRESHOLD,
        )
        if test_std_risk > 0:
            risk_score += test_std_risk
            risk_reasons.append(f"test split std for {metric_name} = {test_std:.3f}")

        test_range_risk = score_threshold_risk(
            test_range,
            warning_threshold=TEST_RANGE_WARNING_THRESHOLD,
            fail_threshold=TEST_RANGE_FAIL_THRESHOLD,
        )
        if test_range_risk > 0:
            risk_score += test_range_risk
            risk_reasons.append(f"test split range for {metric_name} = {test_range:.3f}")

        cv_std_risk = score_threshold_risk(
            cv_std,
            warning_threshold=CV_STD_WARNING_THRESHOLD,
            fail_threshold=CV_STD_FAIL_THRESHOLD,
        )
        if cv_std_risk > 0:
            risk_score += cv_std_risk
            risk_reasons.append(f"cv_oof std for {metric_name} = {cv_std:.3f}")

        if calibration_status == "weak":
            risk_score += 1
            risk_reasons.append(f"weak calibration by test brier = {test_brier_mean:.3f}")

        secondary_risk_score, secondary_risk_reasons = collect_secondary_gap_risk(
            gap_diagnostics_df,
            model_name=model_name,
            primary_metric=metric_name,
        )
        risk_score += secondary_risk_score
        risk_reasons.extend(secondary_risk_reasons)

        rows.append(
            {
                "validation_protocol_name": protocol.name,
                "benchmark_protocol_name": protocol.comparison_protocol.name,
                "model_name": model_name,
                "audit_metric": metric_name,
                "precision_k": int(test_row["precision_k"]),
                "train_in_sample_mean": scalar_to_float(train_row["score_mean"]),
                "cv_oof_mean": (
                    scalar_to_float(cv_row["score_mean"])
                    if cv_row is not None
                    else float("nan")
                ),
                "test_holdout_mean": scalar_to_float(test_row["score_mean"]),
                "test_holdout_std": test_std,
                "test_holdout_min": scalar_to_float(test_row["score_min"]),
                "test_holdout_max": scalar_to_float(test_row["score_max"]),
                "avg_train_test_gap": avg_gap,
                "max_train_test_gap": max_gap,
                "avg_cv_test_gap": cv_gap_mean,
                "cv_oof_std": cv_std,
                "test_brier_mean": test_brier_mean,
                "calibration_status": calibration_status,
                "risk_score": int(risk_score),
                "risk_level": risk_level_from_score(risk_score),
                "risk_reasons": "; ".join(risk_reasons) if risk_reasons else "",
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        ["risk_score", "model_name"],
        ascending=[False, True],
        ignore_index=True,
    )


__all__ = [
    "AVG_GAP_FAIL_THRESHOLD",
    "AVG_GAP_WARNING_THRESHOLD",
    "BrierCalibrationStatus",
    "CV_STD_FAIL_THRESHOLD",
    "CV_STD_WARNING_THRESHOLD",
    "collect_secondary_gap_risk",
    "LOW_MAX_RISK_SCORE",
    "MAX_GAP_FAIL_THRESHOLD",
    "MAX_GAP_WARNING_THRESHOLD",
    "MODERATE_MAX_RISK_SCORE",
    "TEST_RANGE_FAIL_THRESHOLD",
    "TEST_RANGE_WARNING_THRESHOLD",
    "TEST_STD_FAIL_THRESHOLD",
    "TEST_STD_WARNING_THRESHOLD",
    "build_model_risk_audit_frame",
    "calibration_status_from_brier",
    "risk_level_from_score",
    "score_threshold_risk",
]
