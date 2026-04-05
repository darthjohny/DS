# Файл `quality_gate_calibration_review.py` слоя `reporting`.
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

from exohost.posthoc.quality_gate_tuning import (
    DEFAULT_QUALITY_GATE_TUNING_CONFIG,
    QualityGateTuningConfig,
    apply_quality_gate_tuning,
)

QualityGateCalibrationSpec = QualityGateTuningConfig


DEFAULT_QUALITY_GATE_CALIBRATION_SPECS: tuple[QualityGateCalibrationSpec, ...] = (
    QualityGateCalibrationSpec(
        policy_name=DEFAULT_QUALITY_GATE_TUNING_CONFIG.policy_name,
        ruwe_unknown_threshold=DEFAULT_QUALITY_GATE_TUNING_CONFIG.ruwe_unknown_threshold,
        parallax_snr_unknown_threshold=(
            DEFAULT_QUALITY_GATE_TUNING_CONFIG.parallax_snr_unknown_threshold
        ),
        require_flame_for_pass=DEFAULT_QUALITY_GATE_TUNING_CONFIG.require_flame_for_pass,
    ),
    QualityGateCalibrationSpec(
        policy_name="relaxed",
        ruwe_unknown_threshold=1.6,
        parallax_snr_unknown_threshold=3.0,
        require_flame_for_pass=False,
    ),
    QualityGateCalibrationSpec(
        policy_name="strict",
        ruwe_unknown_threshold=1.2,
        parallax_snr_unknown_threshold=7.0,
        require_flame_for_pass=True,
    ),
)


def build_quality_gate_variant_summary_frame(
    df: pd.DataFrame,
    *,
    specs: tuple[QualityGateCalibrationSpec, ...] = DEFAULT_QUALITY_GATE_CALIBRATION_SPECS,
) -> pd.DataFrame:
    # Сводка coverage по нескольким вариантам quality-gate policy.
    rows: list[dict[str, object]] = []
    total_rows = int(df.shape[0])

    for spec in specs:
        variant_df = _evaluate_quality_gate_variant(df, spec)
        quality_state = _require_series_column(variant_df, "variant_quality_state")
        rows.append(
            {
                "policy_name": spec.policy_name,
                "ruwe_unknown_threshold": (
                    pd.NA
                    if spec.ruwe_unknown_threshold is None
                    else float(spec.ruwe_unknown_threshold)
                ),
                "parallax_snr_unknown_threshold": (
                    pd.NA
                    if spec.parallax_snr_unknown_threshold is None
                    else float(spec.parallax_snr_unknown_threshold)
                ),
                "require_flame_for_pass": bool(spec.require_flame_for_pass),
                "n_rows": total_rows,
                "n_pass_rows": int((quality_state == "pass").sum()),
                "n_unknown_rows": int((quality_state == "unknown").sum()),
                "n_reject_rows": int((quality_state == "reject").sum()),
                "share_pass": float((quality_state == "pass").mean()) if total_rows > 0 else 0.0,
                "share_unknown": float((quality_state == "unknown").mean())
                if total_rows > 0
                else 0.0,
                "share_reject": float((quality_state == "reject").mean())
                if total_rows > 0
                else 0.0,
            }
        )

    return pd.DataFrame.from_records(rows)


def build_quality_gate_variant_transition_frame(
    df: pd.DataFrame,
    *,
    spec: QualityGateCalibrationSpec,
) -> pd.DataFrame:
    # Сравниваем live quality_state и variant quality_state по crosstab.
    variant_df = _evaluate_quality_gate_variant(df, spec)
    current_quality_state = _require_series_column(variant_df, "quality_state")
    variant_quality_state = _require_series_column(variant_df, "variant_quality_state")
    return pd.crosstab(current_quality_state, variant_quality_state, dropna=False)


def build_quality_gate_variant_reason_frame(
    df: pd.DataFrame,
    *,
    spec: QualityGateCalibrationSpec,
    top_n: int = 20,
) -> pd.DataFrame:
    # Top variant reasons для выбранной policy.
    variant_df = _evaluate_quality_gate_variant(df, spec)
    counts = (
        _require_series_column(variant_df, "variant_quality_reason")
        .astype(str)
        .value_counts(dropna=False)
    )
    total_rows = int(counts.sum())
    rows = [
        {
            "variant_quality_reason": str(label_value),
            "n_rows": int(n_rows),
            "share": float(n_rows / total_rows),
        }
        for label_value, n_rows in counts.items()
    ]
    return pd.DataFrame.from_records(rows).head(top_n).copy()


def build_quality_gate_variant_changed_rows_frame(
    df: pd.DataFrame,
    *,
    spec: QualityGateCalibrationSpec,
    top_n: int = 50,
) -> pd.DataFrame:
    # Показываем объекты, для которых variant policy меняет исход gate state.
    variant_df = _evaluate_quality_gate_variant(df, spec)
    changed_df = variant_df.loc[
        variant_df["quality_state"] != variant_df["variant_quality_state"],
        :,
    ].copy()
    preview_columns = [
        column_name
        for column_name in (
            "source_id",
            "quality_state",
            "variant_quality_state",
            "quality_reason",
            "variant_quality_reason",
            "review_bucket",
            "ruwe",
            "parallax_over_error",
            "radius_flame",
            "non_single_star",
            "classprob_dsc_combmod_star",
            "spectral_class",
            "spectral_subclass",
        )
        if column_name in changed_df.columns
    ]
    return changed_df.loc[:, preview_columns].head(top_n).copy()


def _evaluate_quality_gate_variant(
    df: pd.DataFrame,
    spec: QualityGateCalibrationSpec,
) -> pd.DataFrame:
    tuned_df = apply_quality_gate_tuning(df, config=spec)
    result = df.copy()
    result["variant_quality_state"] = tuned_df["quality_state"]
    result["variant_quality_reason"] = tuned_df["quality_reason"]
    result["variant_review_bucket"] = tuned_df["review_bucket"]
    return result


def _require_series_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column


def _require_numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    numeric_column = pd.to_numeric(_require_series_column(df, column_name), errors="coerce")
    if not isinstance(numeric_column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a numeric pandas Series.")
    return numeric_column
