# Тестовый файл `test_host_priority_review.py` домена `reporting`.
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

from exohost.reporting.host_priority_review import (
    build_host_priority_contract_summary_frame,
    build_host_priority_feature_coverage_frame,
    build_host_priority_missing_core_frame,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _require_float_scalar(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}.")
    return float(value)


def _build_host_priority_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "spec_class": ["G", "K", "M"],
            "evolution_stage": ["dwarf", "evolved", "dwarf"],
            "spec_subclass": ["G2", pd.NA, "M3"],
            "teff_gspphot": [5800.0, 4700.0, 3600.0],
            "logg_gspphot": [4.4, 3.1, 4.9],
            "mh_gspphot": [0.1, -0.2, 0.0],
            "bp_rp": [0.8, 1.2, 2.0],
            "parallax": [12.0, 8.0, 5.0],
            "parallax_over_error": [20.0, 9.0, 6.0],
            "ruwe": [1.0, 1.1, 1.2],
            "radius_flame": [1.0, pd.NA, 0.6],
            "phot_g_mean_mag": [11.0, 12.5, 14.0],
            "radius_gspphot": [1.1, 4.3, 0.7],
        }
    )


def test_build_host_priority_contract_summary_frame_reports_clean_readiness() -> None:
    summary_df = build_host_priority_contract_summary_frame(_build_host_priority_df())

    assert _require_int_scalar(summary_df.loc[0, "n_rows"]) == 3
    assert _require_int_scalar(summary_df.loc[0, "n_core_features"]) == 8
    assert _require_int_scalar(summary_df.loc[0, "n_core_features_present"]) == 8
    assert bool(summary_df.loc[0, "has_canonical_radius_column"]) is True
    assert _require_int_scalar(summary_df.loc[0, "n_rows_with_canonical_radius"]) == 2
    assert _require_int_scalar(summary_df.loc[0, "n_rows_clean_core_ready"]) == 2


def test_host_priority_feature_coverage_frame_tracks_groups_and_presence() -> None:
    coverage_df = build_host_priority_feature_coverage_frame(_build_host_priority_df())

    radius_row = coverage_df.loc[coverage_df["feature_name"] == "radius_flame"].iloc[0]
    phot_row = coverage_df.loc[coverage_df["feature_name"] == "phot_g_mean_mag"].iloc[0]

    assert radius_row["feature_group"] == "core"
    assert bool(radius_row["is_required"]) is True
    assert bool(radius_row["column_present"]) is True
    assert _require_int_scalar(radius_row["n_non_missing"]) == 2
    assert phot_row["feature_group"] == "observability"


def test_build_host_priority_missing_core_frame_highlights_missing_radius() -> None:
    missing_core_df = build_host_priority_missing_core_frame(_build_host_priority_df())

    first_row = missing_core_df.iloc[0]
    assert first_row["feature_name"] == "radius_flame"
    assert _require_int_scalar(first_row["n_missing_rows"]) == 1
    assert _require_float_scalar(first_row["share_missing_rows"]) == (1 / 3)
