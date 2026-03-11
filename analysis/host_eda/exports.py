"""Сохранение артефактов и графиков host EDA."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from analysis.host_eda.constants import OUTPUT_DIR, PLOTS_DIR
from analysis.visual_theme import EXPORT_DPI


def save_plot(filename: str, figure: Figure) -> None:
    """Сохранить matplotlib figure в каталог графиков host EDA.

    Побочные эффекты
    ----------------
    Создаёт каталог `PLOTS_DIR`, сохраняет файл и закрывает figure.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = PLOTS_DIR / filename
    figure_any: Any = figure
    plt_any: Any = plt
    figure_any.tight_layout()
    figure_any.savefig(path, dpi=EXPORT_DPI)
    plt_any.close(figure)


def save_frame(df_part: pd.DataFrame, path: Path) -> None:
    """Сохранить DataFrame в настроенное дерево артефактов host EDA."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_part.to_csv(path, index=False)
