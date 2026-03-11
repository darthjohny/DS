"""Тесты для posterior-aware router EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from router_eda import calc_router_gauss_stats, ensure_router_labels

ROUTER_EDA_COLUMNS = [
    "source_id",
    "spec_class",
    "evolution_stage",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
]


def build_router_eda_df() -> pd.DataFrame:
    """Собрать компактный набор для readiness-проверок."""
    rows = [
        (1, "M", "dwarf", 3450.0, 4.85, 0.42),
        (2, "M", "dwarf", 3520.0, 4.78, 0.45),
        (3, "M", "dwarf", 3380.0, 4.92, 0.40),
        (4, "M", "dwarf", 3495.0, 4.80, 0.43),
        (5, "M", "dwarf", 3425.0, 4.88, 0.41),
        (6, "A", "evolved", 8600.0, 3.20, 3.80),
        (7, "A", "evolved", 8450.0, 3.05, 3.60),
        (8, "A", "evolved", 8750.0, 3.30, 4.00),
        (9, "A", "evolved", 8525.0, 3.18, 3.74),
        (10, "A", "evolved", 8680.0, 3.26, 3.91),
    ]
    frame = pd.DataFrame.from_records(rows, columns=ROUTER_EDA_COLUMNS)
    return ensure_router_labels(frame)


def test_calc_router_gauss_stats_adds_effective_covariance_metrics() -> None:
    """Readiness должен включать posterior-aware поля router-модели."""
    stats = calc_router_gauss_stats(build_router_eda_df())

    assert {
        "shrink_alpha",
        "score_mode",
        "prior_mode",
        "log_det_cov",
        "cond_effective_cov",
        "pd_effective_cov",
        "self_log_posterior_mean",
        "self_posterior_margin_mean",
        "self_posterior_margin_min",
        "self_posterior_win_rate",
    }.issubset(stats.columns)
    assert np.isfinite(stats["log_det_cov"]).all()
    assert np.isfinite(stats["cond_effective_cov"]).all()
    assert stats["pd_effective_cov"].all()
    assert np.isfinite(stats["self_posterior_win_rate"]).all()
