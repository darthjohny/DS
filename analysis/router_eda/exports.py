"""Сохранение артефактов и графиков router EDA."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from analysis.router_eda.constants import (
    CLASS_COUNTS_PATH,
    GAUSS_STATS_PATH,
    OUTPUT_DIR,
    PLOTS_DIR,
    SNAPSHOT_PATH,
)
from analysis.visual_theme import EXPORT_DPI


def save_plot(filename: str, figure: Figure) -> None:
    """Сохранить matplotlib figure в каталог router EDA графиков."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    figure_any: Any = figure
    plt_any: Any = plt
    figure_any.tight_layout()
    figure_any.savefig(path, dpi=EXPORT_DPI)
    plt_any.close(figure)


def save_router_artifacts(
    df_router: pd.DataFrame,
    counts: pd.DataFrame,
    gauss_stats: pd.DataFrame,
) -> None:
    """Сохранить таблицы router EDA для последующего анализа."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_router.to_csv(SNAPSHOT_PATH, index=False)
    counts.to_csv(CLASS_COUNTS_PATH, index=False)
    gauss_stats.to_csv(GAUSS_STATS_PATH, index=False)
