# Тестовый файл `test_quality_gate_review.py` домена `reporting`.
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
from unittest.mock import Mock

import pandas as pd
import pytest

import exohost.reporting.quality_gate_review as quality_gate_review_module
from exohost.reporting.quality_gate_review import (
    build_ood_state_distribution_frame,
    build_quality_gate_signal_summary_frame,
    build_quality_gate_summary_frame,
    build_quality_reason_distribution_frame,
    build_quality_review_crosstab_frame,
    build_quality_state_distribution_frame,
    build_review_bucket_distribution_frame,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _require_float_scalar(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}.")
    return float(value)


def _build_quality_gate_review_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "quality_state": ["pass", "unknown", "reject", "pass"],
            "ood_state": ["in_domain", "candidate_ood", "in_domain", "ood"],
            "quality_reason": [
                "pass",
                "review_high_ruwe",
                "reject_missing_core_features",
                "pass",
            ],
            "ood_reason": ["in_domain", "candidate_ood", "in_domain", "ood"],
            "review_bucket": [pd.NA, "review_high_ruwe", "reject_missing_core_features", "ood"],
            "has_core_features": [True, True, False, True],
            "has_flame_features": [True, False, False, True],
            "has_non_single_star_flag": [False, False, False, True],
            "has_low_single_star_probability": [False, False, False, True],
            "has_missing_core_features": [False, False, True, False],
            "has_missing_flame_features": [False, True, True, False],
            "has_high_ruwe": [False, True, False, False],
            "has_low_parallax_snr": [False, False, False, True],
            "ruwe": [1.1, 1.7, 1.2, 1.0],
            "parallax_over_error": [10.0, 4.2, 8.0, 2.1],
            "non_single_star": [0, 0, 0, 1],
            "classprob_dsc_combmod_star": [0.9, 0.8, 0.7, 0.2],
            "radius_flame": [1.0, pd.NA, pd.NA, 2.0],
            "random_index": [10, 11, 12, 13],
        }
    )


def test_build_quality_gate_summary_frame_returns_compact_counts() -> None:
    summary_df = build_quality_gate_summary_frame(_build_quality_gate_review_df())

    assert _require_int_scalar(summary_df.loc[0, "n_rows"]) == 4
    assert _require_int_scalar(summary_df.loc[0, "n_unique_source_id"]) == 4
    assert _require_int_scalar(summary_df.loc[0, "n_pass_rows"]) == 2
    assert _require_int_scalar(summary_df.loc[0, "n_unknown_rows"]) == 1
    assert _require_int_scalar(summary_df.loc[0, "n_reject_rows"]) == 1
    assert _require_int_scalar(summary_df.loc[0, "n_candidate_ood_rows"]) == 1
    assert _require_int_scalar(summary_df.loc[0, "n_ood_rows"]) == 1


def test_quality_gate_distribution_helpers_cover_state_and_reason_balance() -> None:
    review_df = _build_quality_gate_review_df()

    quality_state_df = build_quality_state_distribution_frame(review_df)
    ood_state_df = build_ood_state_distribution_frame(review_df)
    quality_reason_df = build_quality_reason_distribution_frame(review_df, top_n=10)
    review_bucket_df = build_review_bucket_distribution_frame(review_df, top_n=10)
    crosstab_df = build_quality_review_crosstab_frame(review_df)

    assert set(quality_state_df["quality_state"]) == {"pass", "unknown", "reject"}
    assert set(ood_state_df["ood_state"]) == {"in_domain", "candidate_ood", "ood"}
    assert "review_high_ruwe" in set(quality_reason_df["quality_reason"])
    assert "review_high_ruwe" in set(review_bucket_df["review_bucket"])
    assert _require_int_scalar(crosstab_df.loc["unknown", "review_high_ruwe"]) == 1


def test_build_quality_gate_signal_summary_frame_counts_signal_impact() -> None:
    signal_df = build_quality_gate_signal_summary_frame(_build_quality_gate_review_df())

    high_ruwe_row = signal_df.loc[
        signal_df["signal_name"] == "has_high_ruwe"
    ].iloc[0]
    missing_core_row = signal_df.loc[
        signal_df["signal_name"] == "has_missing_core_features"
    ].iloc[0]

    assert _require_int_scalar(high_ruwe_row["n_rows_true"]) == 1
    assert _require_int_scalar(high_ruwe_row["n_unknown_true"]) == 1
    assert _require_float_scalar(high_ruwe_row["share_true"]) == 0.25
    assert _require_int_scalar(missing_core_row["n_reject_true"]) == 1


def test_load_quality_gate_review_frame_disposes_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Mock()
    expected_df = pd.DataFrame({"source_id": [1]})

    monkeypatch.setattr(
        quality_gate_review_module,
        "make_read_only_engine",
        lambda *, dotenv_path=".env": engine,
    )
    monkeypatch.setattr(
        quality_gate_review_module,
        "load_quality_gate_audit_dataset",
        lambda current_engine, *, contract, limit=None: expected_df,
    )

    result_df = quality_gate_review_module.load_quality_gate_review_frame()

    assert result_df.equals(expected_df)
    engine.dispose.assert_called_once_with()
