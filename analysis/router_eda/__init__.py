"""Публичный API исследовательского router EDA.

Пакет объединяет:

- загрузку reference-выборки и label helpers;
- readiness-статистики для router-модели;
- plotting/export helpers;
- CLI-точку входа для построения router EDA.
"""

from __future__ import annotations

from analysis.router_eda.cli import main
from analysis.router_eda.constants import (
    CLASS_COUNTS_PATH,
    EVOLUTION_STAGES,
    FEATURES,
    GAUSS_STATS_PATH,
    MIN_READY_CLASS_SIZE,
    OUTPUT_DIR,
    PLOTS_DIR,
    ROUTER_PRIOR_MODE,
    ROUTER_SCORE_MODE,
    ROUTER_SHRINK_ALPHA,
    ROUTER_VIEW,
    SNAPSHOT_PATH,
    SPEC_CLASSES,
    FloatArray,
)
from analysis.router_eda.data import (
    build_class_counts,
    build_router_training_query,
    ensure_router_labels,
    feature_frame,
    make_engine_from_env,
    make_router_label,
    read_sql_frame,
)
from analysis.router_eda.exports import save_plot, save_router_artifacts
from analysis.router_eda.plots import (
    draw_boxplot,
    draw_countplot,
    draw_heatmap,
    draw_histplot,
    draw_scatterplot,
    make_figure_ax,
    make_figure_axes,
    plot_router_correlation_heatmaps,
    plot_router_feature_boxplots,
    plot_router_feature_histograms,
    plot_router_label_counts,
    plot_router_scatter,
    set_axes_title,
    set_axes_xlabel,
    set_axes_ylabel,
    set_default_theme,
)
from analysis.router_eda.readiness import (
    calc_router_gauss_stats,
    has_missing_values,
    is_missing_scalar,
)
from analysis.router_eda.stats import (
    print_basic_summary,
    print_class_counts,
    print_gaussian_readiness,
    print_null_report,
    print_router_describe,
)

__all__ = [
    "CLASS_COUNTS_PATH",
    "EVOLUTION_STAGES",
    "FEATURES",
    "FloatArray",
    "GAUSS_STATS_PATH",
    "MIN_READY_CLASS_SIZE",
    "OUTPUT_DIR",
    "PLOTS_DIR",
    "ROUTER_PRIOR_MODE",
    "ROUTER_SCORE_MODE",
    "ROUTER_SHRINK_ALPHA",
    "ROUTER_VIEW",
    "SNAPSHOT_PATH",
    "SPEC_CLASSES",
    "build_class_counts",
    "build_router_training_query",
    "calc_router_gauss_stats",
    "draw_boxplot",
    "draw_countplot",
    "draw_heatmap",
    "draw_histplot",
    "draw_scatterplot",
    "ensure_router_labels",
    "feature_frame",
    "has_missing_values",
    "is_missing_scalar",
    "main",
    "make_engine_from_env",
    "make_figure_ax",
    "make_figure_axes",
    "make_router_label",
    "plot_router_correlation_heatmaps",
    "plot_router_feature_boxplots",
    "plot_router_feature_histograms",
    "plot_router_label_counts",
    "plot_router_scatter",
    "print_basic_summary",
    "print_class_counts",
    "print_gaussian_readiness",
    "print_null_report",
    "print_router_describe",
    "read_sql_frame",
    "save_plot",
    "save_router_artifacts",
    "set_axes_title",
    "set_axes_xlabel",
    "set_axes_ylabel",
    "set_default_theme",
]
