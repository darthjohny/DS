# Тестовый файл `test_coarse_secure_o_tail_review.py` домена `reporting`.
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

from exohost.reporting.coarse_secure_o_tail_review import (
    build_secure_o_tail_esphs_frame,
    build_secure_o_tail_raw_label_frame,
    build_secure_o_tail_review_query,
    build_secure_o_tail_summary_frame,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def _require_float_scalar(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}.")
    return float(value)


def test_build_secure_o_tail_review_query_uses_secure_subset_and_raw_labels() -> None:
    query = build_secure_o_tail_review_query(limit=10)

    assert 'FROM "lab"."gaia_ob_secure_o_like_subset" AS secure_o' in query
    assert 'JOIN "lab"."gaia_mk_external_labeled" AS labeled' in query
    assert 'WHERE secure_o."spectral_class" = \'O\'' in query
    assert "LIMIT 10" in query


def test_build_secure_o_tail_summary_frame_counts_available_hot_star_signals() -> None:
    review_df = pd.DataFrame(
        {
            "raw_sptype": ["O4V", "O8I"],
            "spectral_subclass": [4.0, pd.NA],
            "teff_gspphot": [15000.0, 11000.0],
            "teff_esphs": [33000.0, pd.NA],
            "lum_flame": [2800.0, 2400.0],
            "in_gold_sample_oba_stars": [True, False],
        }
    )

    summary_df = build_secure_o_tail_summary_frame(review_df)

    assert _require_int_scalar(summary_df.loc[0, "n_rows"]) == 2
    assert _require_int_scalar(summary_df.loc[0, "n_with_numeric_subclass"]) == 1
    assert _require_int_scalar(summary_df.loc[0, "n_with_teff_esphs"]) == 1
    assert _require_int_scalar(summary_df.loc[0, "n_in_gold_sample_oba"]) == 1
    assert _require_float_scalar(summary_df.loc[0, "median_teff_gspphot"]) == 13000.0


def test_build_secure_o_tail_raw_label_and_esphs_frames_keep_counts() -> None:
    review_df = pd.DataFrame(
        {
            "raw_sptype": ["O4V", "O4V", "O8I"],
            "spectraltype_esphs": ["O", "O", "O"],
            "flags_esphs": [1.0, 91.0, 91.0],
            "teff_esphs": [33000.0, pd.NA, pd.NA],
        }
    )

    raw_label_df = build_secure_o_tail_raw_label_frame(review_df)
    esphs_df = build_secure_o_tail_esphs_frame(review_df)

    assert list(raw_label_df["raw_sptype"]) == ["O4V", "O8I"]
    assert list(raw_label_df["n_rows"]) == [2, 1]
    assert _require_int_scalar(esphs_df.loc[:, "n_rows"].sum()) == 3
