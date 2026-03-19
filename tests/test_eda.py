"""Тесты для contrastive host EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis.host_eda.contrastive import calc_contrastive_gauss_stats

EDA_COLUMNS = [
    "spec_class",
    "is_host",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
]


def build_contrastive_eda_df() -> pd.DataFrame:
    """Собрать компактный host-vs-field набор для EDA readiness."""
    rows = [
        ("M", True, 3450.0, 4.85, 0.42),
        ("M", True, 3490.0, 4.82, 0.43),
        ("M", False, 3520.0, 4.78, 0.45),
        ("M", False, 3380.0, 4.92, 0.40),
        ("K", True, 4820.0, 4.63, 0.83),
        ("K", True, 4880.0, 4.68, 0.86),
        ("K", False, 4760.0, 4.58, 0.79),
        ("K", False, 4710.0, 4.55, 0.77),
        ("G", True, 5670.0, 4.44, 1.04),
        ("G", True, 5600.0, 4.39, 0.98),
        ("G", False, 5740.0, 4.50, 1.02),
        ("G", False, 5790.0, 4.54, 1.06),
        ("F", True, 6240.0, 4.28, 1.21),
        ("F", True, 6185.0, 4.24, 1.17),
        ("F", False, 6120.0, 4.19, 1.15),
        ("F", False, 6080.0, 4.14, 1.12),
    ]
    return pd.DataFrame.from_records(rows, columns=EDA_COLUMNS)


def test_calc_contrastive_gauss_stats_adds_host_field_metrics() -> None:
    """Contrastive host EDA должен возвращать host/field readiness поля."""
    stats = calc_contrastive_gauss_stats(
        build_contrastive_eda_df(),
        population_col="is_host",
        use_m_subclasses=False,
        shrink_alpha=0.10,
        min_population_size=2,
    )

    assert {
        "host_n",
        "field_n",
        "host_log_det_cov",
        "field_log_det_cov",
        "host_cond_cov",
        "field_cond_cov",
        "host_pd_cov",
        "field_pd_cov",
        "host_log_lr_mean_host",
        "host_log_lr_mean_field",
        "host_posterior_mean_host",
        "host_posterior_mean_field",
        "contrastive_margin_mean",
        "contrastive_accuracy",
        "centroid_distance_z",
    }.issubset(stats.columns)
    assert np.isfinite(stats["host_log_det_cov"]).all()
    assert np.isfinite(stats["field_log_det_cov"]).all()
    assert np.isfinite(stats["host_cond_cov"]).all()
    assert np.isfinite(stats["field_cond_cov"]).all()
    assert stats["host_pd_cov"].all()
    assert stats["field_pd_cov"].all()
    assert stats["ready_for_contrastive_model"].all()
    assert np.isfinite(stats["contrastive_accuracy"]).all()
