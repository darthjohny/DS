# Тестовый файл `test_quality_gate_calibration_review.py` домена `reporting`.
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

import pandas as pd

from exohost.reporting.quality_gate_calibration_review import (
    QualityGateCalibrationSpec,
    build_quality_gate_variant_changed_rows_frame,
    build_quality_gate_variant_reason_frame,
    build_quality_gate_variant_summary_frame,
    build_quality_gate_variant_transition_frame,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _require_float_scalar(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}.")
    return float(value)


def _build_quality_gate_df() -> pd.DataFrame:
    # В кадре есть базовый pass, review по RUWE, hard reject и review по FLAME.
    # Этого достаточно, чтобы увидеть, как policy-варианты меняют покрытие и причины.
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "quality_state": ["pass", "unknown", "reject", "unknown"],
            "quality_reason": [
                "pass",
                "review_high_ruwe",
                "reject_missing_core_features",
                "review_missing_radius_flame",
            ],
            "review_bucket": [pd.NA, "review_high_ruwe", "reject_missing_core_features", "review_missing_radius_flame"],
            "has_missing_core_features": [False, False, True, False],
            "has_missing_flame_features": [False, False, False, True],
            "ruwe": [1.1, 1.55, 1.0, 1.1],
            "parallax_over_error": [8.0, 6.0, 9.0, 4.0],
            "radius_flame": [1.0, 2.0, 3.0, pd.NA],
            "non_single_star": [0, 0, 0, 0],
            "classprob_dsc_combmod_star": [0.9, 0.8, 0.7, 0.6],
            "spectral_class": ["G", "K", "M", "F"],
            "spectral_subclass": ["G2", "K3", "M1", "F5"],
        }
    )


def test_build_quality_gate_variant_summary_frame_compares_policy_coverage() -> None:
    # Сравниваем базовый и смягченный варианты policy на одном frozen наборе
    # и смотрим, как меняются доли `pass` и `unknown`.
    specs = (
        QualityGateCalibrationSpec(
            policy_name="baseline",
            ruwe_unknown_threshold=1.4,
            parallax_snr_unknown_threshold=5.0,
            require_flame_for_pass=True,
        ),
        QualityGateCalibrationSpec(
            policy_name="relaxed",
            ruwe_unknown_threshold=1.6,
            parallax_snr_unknown_threshold=3.0,
            require_flame_for_pass=False,
        ),
    )

    summary_df = build_quality_gate_variant_summary_frame(
        _build_quality_gate_df(),
        specs=specs,
    )

    baseline_row = summary_df.loc[summary_df["policy_name"] == "baseline"].iloc[0]
    relaxed_row = summary_df.loc[summary_df["policy_name"] == "relaxed"].iloc[0]

    assert _require_int_scalar(baseline_row["n_unknown_rows"]) == 2
    assert _require_int_scalar(relaxed_row["n_unknown_rows"]) == 0
    assert _require_float_scalar(relaxed_row["share_pass"]) == 0.75


def test_variant_transition_and_reason_frames_explain_policy_differences() -> None:
    # Помимо summary, review-слой должен объяснять, какие именно строки и причины
    # меняются при выборе другого варианта `quality_gate`.
    spec = QualityGateCalibrationSpec(
        policy_name="relaxed",
        ruwe_unknown_threshold=1.6,
        parallax_snr_unknown_threshold=3.0,
        require_flame_for_pass=False,
    )
    review_df = _build_quality_gate_df()

    transition_df = build_quality_gate_variant_transition_frame(review_df, spec=spec)
    reason_df = build_quality_gate_variant_reason_frame(review_df, spec=spec, top_n=10)
    changed_rows_df = build_quality_gate_variant_changed_rows_frame(
        review_df,
        spec=spec,
        top_n=10,
    )

    assert _require_int_scalar(transition_df.loc["unknown", "pass"]) == 2
    assert "pass" in set(reason_df["variant_quality_reason"])
    assert list(changed_rows_df["source_id"]) == [2, 4]
