"""Фасад совместимости для исследовательского host EDA.

Что делает модуль:
    - реэкспортирует публичные функции, константы и CLI из
      пакета `analysis.host_eda`;
    - сохраняет старую точку входа `src/eda.py`;
    - позволяет старому коду импортировать host EDA без
      перехода на новый пакетный путь.

Где находится основная логика:
    - загрузка данных: `analysis.host_eda.data`;
    - статистики и contrastive-readiness: `analysis.host_eda.stats`,
      `analysis.host_eda.contrastive`;
    - графики и экспорт артефактов: `analysis.host_eda.plots`,
      `analysis.host_eda.exports`;
    - сценарий запуска: `analysis.host_eda.cli`.

Что модуль не делает:
    - не содержит собственной EDA-логики;
    - не является каноническим местом для новых исследований.
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.host_eda.cli import main
from analysis.host_eda.constants import (
    ABO_TOP20_PATH,
    CLASS_ORDER,
    CONTRASTIVE_GAUSS_STATS_PATH,
    CONTRASTIVE_MIN_POPULATION_SIZE,
    CONTRASTIVE_MODEL_VERSION,
    CONTRASTIVE_SCORE_MODE,
    CONTRASTIVE_SHRINK_ALPHA,
    CONTRASTIVE_VIEW_ENV,
    EVOLVED_SNAPSHOT_PATH,
    FEATURES,
    LOGG_DWARF_MIN,
    M_EARLY_MAX,
    M_EARLY_MIN,
    M_MID_MIN,
    OUTPUT_DIR,
    PLOTS_DIR,
    QUERY_ABO_REF,
    QUERY_ALL_MKGF,
    QUERY_DWARFS_MKGF,
    QUERY_EVOLVED_MKGF,
    FloatArray,
)
from analysis.host_eda.contrastive import (
    calc_contrastive_gauss_stats,
    print_contrastive_gauss_stats,
)
from analysis.host_eda.data import (
    feature_frame,
    get_engine,
    load_abo_ref,
    load_all_mkgf,
    load_dwarfs_mkgf,
    load_evolved_mkgf,
    make_engine_from_env,
    read_sql_frame,
)
from analysis.host_eda.exports import save_frame, save_plot
from analysis.host_eda.plots import (
    draw_axes_legend,
    draw_axes_scatter,
    draw_axes_vline,
    draw_boxplot,
    draw_countplot,
    draw_heatmap,
    draw_histplot,
    draw_scatterplot,
    make_figure_ax,
    make_figure_axes,
    plot_boxplots_dwarfs,
    plot_class_counts,
    plot_correlation_heatmaps,
    plot_feature_histograms,
    plot_logg_radius_with_threshold,
    plot_scatter_layers,
    set_axes_title,
    set_axes_xlabel,
    set_axes_ylabel,
    set_default_theme,
)
from analysis.host_eda.stats import (
    build_group_stats,
    calc_gauss_stats,
    top_by_radius,
    top_by_teff,
)

__all__ = [
    "ABO_TOP20_PATH",
    "CLASS_ORDER",
    "CONTRASTIVE_GAUSS_STATS_PATH",
    "CONTRASTIVE_MIN_POPULATION_SIZE",
    "CONTRASTIVE_MODEL_VERSION",
    "CONTRASTIVE_SCORE_MODE",
    "CONTRASTIVE_SHRINK_ALPHA",
    "CONTRASTIVE_VIEW_ENV",
    "EVOLVED_SNAPSHOT_PATH",
    "FEATURES",
    "FloatArray",
    "LOGG_DWARF_MIN",
    "M_EARLY_MAX",
    "M_EARLY_MIN",
    "M_MID_MIN",
    "OUTPUT_DIR",
    "PLOTS_DIR",
    "QUERY_ABO_REF",
    "QUERY_ALL_MKGF",
    "QUERY_DWARFS_MKGF",
    "QUERY_EVOLVED_MKGF",
    "build_group_stats",
    "calc_contrastive_gauss_stats",
    "calc_gauss_stats",
    "draw_axes_legend",
    "draw_axes_scatter",
    "draw_axes_vline",
    "draw_boxplot",
    "draw_countplot",
    "draw_heatmap",
    "draw_histplot",
    "draw_scatterplot",
    "feature_frame",
    "get_engine",
    "load_abo_ref",
    "load_all_mkgf",
    "load_dwarfs_mkgf",
    "load_evolved_mkgf",
    "main",
    "make_engine_from_env",
    "make_figure_ax",
    "make_figure_axes",
    "plot_boxplots_dwarfs",
    "plot_class_counts",
    "plot_correlation_heatmaps",
    "plot_feature_histograms",
    "plot_logg_radius_with_threshold",
    "plot_scatter_layers",
    "print_contrastive_gauss_stats",
    "read_sql_frame",
    "save_frame",
    "save_plot",
    "set_axes_title",
    "set_axes_xlabel",
    "set_axes_ylabel",
    "set_default_theme",
    "top_by_radius",
    "top_by_teff",
]


if __name__ == "__main__":
    main()
