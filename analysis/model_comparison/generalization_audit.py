"""Per-model generalization audit для comparison-layer.

Модуль не обучает модели заново. Он собирает диагностический verdict по
каждой модели на основе уже посчитанных benchmark-артефактов:

- общей metrics summary;
- class-wise test metrics;
- generalization diagnostics с train/test и CV/test gap.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from analysis.model_comparison.contracts import DEFAULT_COMPARISON_PROTOCOL, ComparisonProtocol

PASS_MAX_RISK_SCORE = 0
WATCH_MAX_RISK_SCORE = 2
GAP_WARNING_THRESHOLD = 0.05
GAP_FAIL_THRESHOLD = 0.10
CV_STD_WARNING_THRESHOLD = 0.04
CV_STD_FAIL_THRESHOLD = 0.08
CLASSWISE_RANGE_WARNING_THRESHOLD = 0.10
CLASSWISE_RANGE_FAIL_THRESHOLD = 0.20
BrierCalibrationStatus = str


def score_threshold_risk(
    value: float,
    *,
    warning_threshold: float,
    fail_threshold: float,
) -> int:
    """Преобразовать диагностическое значение в risk points."""
    if value > fail_threshold:
        return 2
    if value > warning_threshold:
        return 1
    return 0


def calibration_status_from_brier(test_brier: float) -> BrierCalibrationStatus:
    """Собрать coarse calibration verdict по test Brier score."""
    if test_brier <= 0.10:
        return "good"
    if test_brier <= 0.20:
        return "watch"
    return "weak"


def risk_level_from_score(risk_score: int) -> str:
    """Преобразовать integer risk score в человекочитаемый verdict."""
    if risk_score <= PASS_MAX_RISK_SCORE:
        return "PASS"
    if risk_score <= WATCH_MAX_RISK_SCORE:
        return "WATCH"
    return "FAIL"


def build_model_generalization_audit_frame(
    summary_df: pd.DataFrame,
    classwise_df: pd.DataFrame,
    generalization_df: pd.DataFrame,
) -> pd.DataFrame:
    """Собрать per-model generalization audit summary."""
    if summary_df.empty or classwise_df.empty or generalization_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for model_name, group in summary_df.groupby("model_name", sort=True):
        split_lookup = {
            str(row["split_name"]): row for _, row in group.iterrows()
        }
        test_row = split_lookup.get("test")
        if test_row is None:
            raise ValueError(
                "Generalization audit requires a test summary row "
                f"for model {model_name}."
            )

        model_generalization_df = generalization_df[
            generalization_df["model_name"] == model_name
        ].copy()
        if model_generalization_df.empty:
            raise ValueError(
                "Generalization audit requires diagnostics rows "
                f"for model {model_name}."
            )

        refit_rows = model_generalization_df[
            model_generalization_df["is_refit_metric"].astype(bool)
        ]
        audit_scope = "refit_metric"
        if refit_rows.empty:
            refit_rows = model_generalization_df[
                model_generalization_df["metric_name"] == "roc_auc"
            ]
            audit_scope = "fallback_roc_auc"
        if refit_rows.empty:
            raise ValueError(
                "Generalization audit requires either a refit metric row "
                f"or roc_auc fallback for model {model_name}."
            )

        refit_row = refit_rows.iloc[0]
        refit_metric = str(refit_row["metric_name"])
        classwise_test_df = classwise_df[
            (classwise_df["model_name"] == model_name)
            & (classwise_df["split_name"] == "test")
        ].copy()
        if classwise_test_df.empty:
            raise ValueError(
                "Generalization audit requires test class-wise rows "
                f"for model {model_name}."
            )

        classwise_metric = classwise_test_df[refit_metric].astype(float)
        test_brier = float(test_row["brier"])
        calibration_status = calibration_status_from_brier(test_brier)

        abs_train_test_gap = abs(float(refit_row["train_minus_test"]))
        cv_minus_test = float(refit_row["cv_minus_test"])
        abs_cv_test_gap = abs(cv_minus_test) if pd.notna(cv_minus_test) else float("nan")
        cv_score_std = float(refit_row["cv_score_std"]) if pd.notna(refit_row["cv_score_std"]) else float("nan")
        classwise_range = float(classwise_metric.max() - classwise_metric.min())
        classwise_std = float(classwise_metric.std(ddof=0))

        risk_score = 0
        risk_reasons: list[str] = []

        gap_risk = score_threshold_risk(
            abs_train_test_gap,
            warning_threshold=GAP_WARNING_THRESHOLD,
            fail_threshold=GAP_FAIL_THRESHOLD,
        )
        if gap_risk > 0:
            risk_score += gap_risk
            risk_reasons.append(
                f"train/test gap for {refit_metric} = {abs_train_test_gap:.3f}"
            )

        if pd.notna(abs_cv_test_gap):
            cv_gap_risk = score_threshold_risk(
                abs_cv_test_gap,
                warning_threshold=GAP_WARNING_THRESHOLD,
                fail_threshold=GAP_FAIL_THRESHOLD,
            )
            if cv_gap_risk > 0:
                risk_score += cv_gap_risk
                risk_reasons.append(
                    f"cv/test gap for {refit_metric} = {abs_cv_test_gap:.3f}"
                )

        if pd.notna(cv_score_std):
            cv_std_risk = score_threshold_risk(
                cv_score_std,
                warning_threshold=CV_STD_WARNING_THRESHOLD,
                fail_threshold=CV_STD_FAIL_THRESHOLD,
            )
            if cv_std_risk > 0:
                risk_score += cv_std_risk
                risk_reasons.append(
                    f"cv std for {refit_metric} = {cv_score_std:.3f}"
                )

        classwise_risk = score_threshold_risk(
            classwise_range,
            warning_threshold=CLASSWISE_RANGE_WARNING_THRESHOLD,
            fail_threshold=CLASSWISE_RANGE_FAIL_THRESHOLD,
        )
        if classwise_risk > 0:
            risk_score += classwise_risk
            risk_reasons.append(
                f"class-wise test range for {refit_metric} = {classwise_range:.3f}"
            )

        if calibration_status == "weak":
            risk_score += 1
            risk_reasons.append(f"weak calibration by test brier = {test_brier:.3f}")

        rows.append(
            {
                "model_name": str(model_name),
                "audit_scope": audit_scope,
                "refit_metric": refit_metric,
                "train_refit_score": float(refit_row["train_value"]),
                "test_refit_score": float(refit_row["test_value"]),
                "train_minus_test": float(refit_row["train_minus_test"]),
                "abs_train_test_gap": abs_train_test_gap,
                "cv_score_mean": (
                    float(refit_row["cv_score_mean"])
                    if pd.notna(refit_row["cv_score_mean"])
                    else float("nan")
                ),
                "cv_score_std": cv_score_std,
                "cv_score_min": (
                    float(refit_row["cv_score_min"])
                    if pd.notna(refit_row["cv_score_min"])
                    else float("nan")
                ),
                "cv_score_max": (
                    float(refit_row["cv_score_max"])
                    if pd.notna(refit_row["cv_score_max"])
                    else float("nan")
                ),
                "cv_minus_test": cv_minus_test,
                "abs_cv_test_gap": abs_cv_test_gap,
                "test_classwise_min": float(classwise_metric.min()),
                "test_classwise_max": float(classwise_metric.max()),
                "test_classwise_range": classwise_range,
                "test_classwise_std": classwise_std,
                "test_brier": test_brier,
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


def frame_to_text(df: pd.DataFrame) -> str:
    """Преобразовать DataFrame в компактный текстовый блок для markdown."""
    if df.empty:
        return "Пусто"
    return df.to_string(index=False)


def build_generalization_audit_markdown(
    audit_df: pd.DataFrame,
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
    note: str = "",
) -> str:
    """Собрать markdown summary для per-model generalization audit."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    note_text = note.strip() or "-"
    return f"""# Generalization Audit Report

Дата: {created_at}
Protocol: `{protocol.name}`

## Per-model verdict
{frame_to_text(audit_df)}

## Примечание
{note_text}
"""


__all__ = [
    "BrierCalibrationStatus",
    "CLASSWISE_RANGE_FAIL_THRESHOLD",
    "CLASSWISE_RANGE_WARNING_THRESHOLD",
    "CV_STD_FAIL_THRESHOLD",
    "CV_STD_WARNING_THRESHOLD",
    "GAP_FAIL_THRESHOLD",
    "GAP_WARNING_THRESHOLD",
    "PASS_MAX_RISK_SCORE",
    "WATCH_MAX_RISK_SCORE",
    "build_generalization_audit_markdown",
    "build_model_generalization_audit_frame",
    "calibration_status_from_brier",
    "risk_level_from_score",
    "score_threshold_risk",
]
