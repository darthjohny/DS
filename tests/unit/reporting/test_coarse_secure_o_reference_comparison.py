# Тестовый файл `test_coarse_secure_o_reference_comparison.py` домена `reporting`.
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

from exohost.reporting.coarse_secure_o_reference_comparison import (
    build_secure_o_reference_comparison_query,
    build_secure_o_reference_distance_frame,
    build_secure_o_reference_group_summary_frame,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _require_float_scalar(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}.")
    return float(value)


def test_build_secure_o_reference_comparison_query_uses_all_reference_groups() -> None:
    query = build_secure_o_reference_comparison_query()

    assert '"lab"."gaia_ob_secure_o_like_subset"' in query
    assert '"public"."gaia_ref_class_o"' in query
    assert '"public"."gaia_ref_class_b"' in query
    assert '"public"."gaia_ref_evolved_class_o"' in query
    assert '"public"."gaia_ref_evolved_class_b"' in query


def test_build_secure_o_reference_group_summary_frame_builds_group_medians() -> None:
    comparison_df = pd.DataFrame(
        {
            "comparison_group": ["secure_o_tail", "secure_o_tail", "reference_b"],
            "teff_gspphot": [15000.0, 16000.0, 13000.0],
            "logg_gspphot": [4.0, 3.8, 4.2],
            "radius_gspphot": [5.0, 6.0, 2.0],
            "parallax": [1.0, 1.4, 2.2],
        }
    )

    summary_df = build_secure_o_reference_group_summary_frame(comparison_df)

    secure_row = summary_df.loc[
        summary_df["comparison_group"] == "secure_o_tail"
    ].reset_index(drop=True)
    assert _require_int_scalar(secure_row.loc[0, "n_rows"]) == 2
    assert _require_float_scalar(secure_row.loc[0, "median_teff_gspphot"]) == 15500.0


def test_build_secure_o_reference_distance_frame_sorts_nearest_group_first() -> None:
    comparison_df = pd.DataFrame(
        {
            "comparison_group": [
                "secure_o_tail",
                "secure_o_tail",
                "reference_b",
                "reference_b",
                "reference_o",
                "reference_o",
            ],
            "teff_gspphot": [15000.0, 16000.0, 13000.0, 12500.0, 30000.0, 31000.0],
            "logg_gspphot": [4.0, 3.8, 4.2, 4.1, 4.0, 3.9],
            "radius_gspphot": [5.0, 6.0, 2.0, 2.5, 7.5, 8.0],
            "parallax": [1.0, 1.4, 2.2, 2.0, 0.4, 0.3],
        }
    )

    distance_df = build_secure_o_reference_distance_frame(comparison_df)

    assert list(distance_df["comparison_group"]) == ["reference_b", "reference_o"]
