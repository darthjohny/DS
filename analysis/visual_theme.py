"""Единая визуальная тема для EDA-графиков и ноутбуков."""

from __future__ import annotations

from typing import Any

import seaborn as sns
from matplotlib.axes import Axes

FIGURE_FACECOLOR = "#faf9f7"
AXES_FACECOLOR = "#f5f3ef"
GRID_COLOR = "#d8d4cc"
SPINE_COLOR = "#bdb7ad"
TEXT_COLOR = "#222222"
LEGEND_FACECOLOR = "#f7f5f1"

FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 11
TITLE_FONT_SIZE = 13
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
LEGEND_TITLE_FONT_SIZE = 10
NOTEBOOK_DPI = 120
EXPORT_DPI = 220

FIGSIZE_COUNT = (7.5, 4.5)
FIGSIZE_STANDARD = (8.5, 5.2)
FIGSIZE_WIDE = (9.0, 5.5)
FIGSIZE_BOX = (9.5, 5.2)
FIGSIZE_HEATMAP_TRIPTYCH = (15.8, 4.8)
FIGSIZE_NOTEBOOK_DOUBLE = (14.0, 4.8)

SPEC_CLASS_COLORS: dict[str, str] = {
    "M": "#c63d2f",
    "K": "#e67e22",
    "G": "#d4ac0d",
    "F": "#9ecae1",
    "A": "#d6eef8",
    "B": "#5dade2",
    "O": "#2e86c1",
}

EVOLUTION_STAGE_COLORS: dict[str, str] = {
    "dwarf": "#6f7d8c",
    "evolved": "#8a5a44",
}

THEME_RC: dict[str, Any] = {
    "figure.facecolor": FIGURE_FACECOLOR,
    "figure.dpi": NOTEBOOK_DPI,
    "axes.facecolor": AXES_FACECOLOR,
    "axes.edgecolor": SPINE_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.titleweight": "semibold",
    "axes.labelsize": LABEL_FONT_SIZE,
    "font.family": FONT_FAMILY,
    "font.size": BASE_FONT_SIZE,
    "xtick.color": TEXT_COLOR,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.color": TEXT_COLOR,
    "ytick.labelsize": TICK_FONT_SIZE,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.85,
    "text.color": TEXT_COLOR,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "legend.facecolor": LEGEND_FACECOLOR,
    "legend.edgecolor": SPINE_COLOR,
    "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
    "savefig.dpi": EXPORT_DPI,
}


def set_project_theme() -> None:
    """Применить общую визуальную тему проекта к аналитическим графикам."""
    sns.set_theme(style="whitegrid", rc=THEME_RC)


def spectral_class_color(label: str) -> str:
    """Вернуть цвет для спектрального класса или составного gauss label."""
    spec_class = str(label).strip().upper().split("_", 1)[0]
    return SPEC_CLASS_COLORS.get(spec_class, "#7f8c8d")


def style_axes(ax: Axes) -> None:
    """Применить общие настройки осей после построения графика."""
    ax_any: Any = ax
    for spine in ax_any.spines.values():
        spine.set_color(SPINE_COLOR)
    ax_any.tick_params(colors=TEXT_COLOR)
