"""Сводки и печать статистики для router EDA."""

from __future__ import annotations

import pandas as pd

from analysis.router_eda.constants import FEATURES
from analysis.router_eda.data import feature_frame


def print_basic_summary(df_router: pd.DataFrame) -> None:
    """Напечатать базовую sanity-check сводку по router dataset."""
    print("\n=== ROUTER DATA LOADED ===")
    print("Размер router dataset:", df_router.shape)
    print(
        "Спектральные классы:",
        sorted(df_router["spec_class"].unique().tolist()),
    )
    print(
        "Эволюционные стадии:",
        sorted(df_router["evolution_stage"].unique().tolist()),
    )


def print_null_report(df_router: pd.DataFrame) -> None:
    """Напечатать отчёт по NULL-значениям в router dataset."""
    columns = ["source_id", "spec_class", "evolution_stage"] + FEATURES
    print("\n=== NULL REPORT ===")
    print(df_router[columns].isnull().sum())


def print_router_describe(df_router: pd.DataFrame) -> None:
    """Напечатать описательные статистики по базовым router-признакам."""
    print("\n=== DESCRIBE (ROUTER FEATURES) ===")
    print(feature_frame(df_router).describe())


def print_class_counts(counts: pd.DataFrame) -> None:
    """Напечатать распределение объектов по router-классам."""
    print("\n=== CLASS COUNTS ===")
    print(counts.to_string(index=False))


def print_gaussian_readiness(stats: pd.DataFrame) -> None:
    """Напечатать компактную сводку Gaussian readiness."""
    print("\n=== GAUSSIAN READINESS ===")
    preview_cols = [
        "router_label",
        "n_objects",
        "log_det_cov",
        "cond_effective_cov",
        "pd_effective_cov",
        "self_posterior_win_rate",
        "self_posterior_margin_mean",
    ]
    print(stats[preview_cols].to_string(index=False))
