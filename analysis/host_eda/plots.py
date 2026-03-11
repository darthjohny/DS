"""Функции построения графиков для host EDA."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis.host_eda.constants import (
    CLASS_ORDER,
    FEATURE_LABELS,
    FEATURES,
    LAYER_LABELS,
    LOGG_DWARF_MIN,
)
from analysis.host_eda.data import feature_frame
from analysis.host_eda.exports import save_plot
from analysis.visual_theme import (
    FIGSIZE_BOX,
    FIGSIZE_COUNT,
    FIGSIZE_HEATMAP_TRIPTYCH,
    FIGSIZE_STANDARD,
    SPEC_CLASS_COLORS,
    set_project_theme,
    style_axes,
)


def set_default_theme() -> None:
    """Применить базовую визуальную тему для host EDA-графиков."""
    set_project_theme()


def make_figure_ax(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Создать типизированную matplotlib figure с одной осью."""
    plt_any: Any = plt
    figure, ax = plt_any.subplots(figsize=figsize)
    return figure, ax


def make_figure_axes(
    ncols: int,
    figsize: tuple[float, float],
) -> tuple[Figure, list[Axes]]:
    """Создать figure и плоский список типизированных осей."""
    plt_any: Any = plt
    figure, axes = plt_any.subplots(1, ncols, figsize=figsize)
    axes_seq = np.atleast_1d(axes)
    typed_axes: list[Axes] = []
    for axis in axes_seq:
        typed_axes.append(axis)
    return figure, typed_axes


def draw_countplot(**kwargs: Any) -> None:
    """Типизированный адаптер для `seaborn.countplot`."""
    sns_any: Any = sns
    sns_any.countplot(**kwargs)


def draw_histplot(**kwargs: Any) -> None:
    """Типизированный адаптер для `seaborn.histplot`."""
    sns_any: Any = sns
    sns_any.histplot(**kwargs)


def draw_boxplot(**kwargs: Any) -> None:
    """Типизированный адаптер для `seaborn.boxplot`."""
    sns_any: Any = sns
    sns_any.boxplot(**kwargs)


def draw_scatterplot(**kwargs: Any) -> None:
    """Типизированный адаптер для `seaborn.scatterplot`."""
    sns_any: Any = sns
    sns_any.scatterplot(**kwargs)


def draw_heatmap(**kwargs: Any) -> None:
    """Типизированный адаптер для `seaborn.heatmap`."""
    sns_any: Any = sns
    sns_any.heatmap(**kwargs)


def set_axes_title(ax: Axes, title: str) -> None:
    """Типизированный адаптер для `Axes.set_title`."""
    ax_any: Any = ax
    ax_any.set_title(title)


def set_axes_xlabel(ax: Axes, label: str) -> None:
    """Типизированный адаптер для `Axes.set_xlabel`."""
    ax_any: Any = ax
    ax_any.set_xlabel(label)


def set_axes_ylabel(ax: Axes, label: str) -> None:
    """Типизированный адаптер для `Axes.set_ylabel`."""
    ax_any: Any = ax
    ax_any.set_ylabel(label)


def draw_axes_scatter(
    ax: Axes,
    x: pd.Series,
    y: pd.Series,
    *,
    s: float,
    alpha: float,
    label: str,
) -> None:
    """Типизированный адаптер для `Axes.scatter`."""
    ax_any: Any = ax
    ax_any.scatter(x, y, s=s, alpha=alpha, label=label)


def draw_axes_vline(
    ax: Axes,
    *,
    x: float,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    """Типизированный адаптер для `Axes.axvline`."""
    ax_any: Any = ax
    ax_any.axvline(
        x=x,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )


def draw_axes_legend(ax: Axes) -> None:
    """Типизированный адаптер для `Axes.legend`."""
    ax_any: Any = ax
    ax_any.legend()


def feature_label(feature: str) -> str:
    """Вернуть локализованное отображаемое имя признака."""
    return FEATURE_LABELS.get(feature, feature)


def layer_label(layer_key: str) -> str:
    """Вернуть локализованное отображаемое имя слоя данных."""
    return LAYER_LABELS.get(layer_key, layer_key)


def plot_class_counts(
    df_part: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    """Построить график числа объектов по спектральным классам для слоя."""
    figure, ax = make_figure_ax(FIGSIZE_COUNT)
    draw_countplot(
        data=df_part,
        x="spec_class",
        hue="spec_class",
        order=CLASS_ORDER,
        hue_order=CLASS_ORDER,
        palette=SPEC_CLASS_COLORS,
        legend=False,
        ax=ax,
    )
    set_axes_title(ax, title)
    set_axes_xlabel(ax, "Спектральный класс")
    set_axes_ylabel(ax, "Число объектов")
    style_axes(ax)
    save_plot(filename, figure)


def plot_feature_histograms(
    df_part: pd.DataFrame,
    filename_prefix: str,
    title_prefix: str,
) -> None:
    """Построить гистограммы признаков с разбиением по спектральному классу."""
    for feature in FEATURES:
        figure, ax = make_figure_ax(FIGSIZE_STANDARD)
        draw_histplot(
            data=df_part,
            x=feature,
            hue="spec_class",
            hue_order=CLASS_ORDER,
            palette=SPEC_CLASS_COLORS,
            bins=30,
            stat="density",
            common_norm=False,
            alpha=0.35,
            kde=True,
            ax=ax,
        )
        set_axes_title(
            ax,
            f"{title_prefix}: распределение признака «{feature_label(feature)}»",
        )
        set_axes_xlabel(ax, feature_label(feature))
        set_axes_ylabel(ax, "Плотность")
        style_axes(ax)
        save_plot(f"{filename_prefix}_{feature}_hist.png", figure)


def plot_boxplots_dwarfs(df_part: pd.DataFrame) -> None:
    """Построить boxplot-графики для выборки карликов по классам."""
    for feature in FEATURES:
        figure, ax = make_figure_ax(FIGSIZE_BOX)
        draw_boxplot(
            data=df_part,
            x="spec_class",
            y=feature,
            hue="spec_class",
            order=CLASS_ORDER,
            hue_order=CLASS_ORDER,
            palette=SPEC_CLASS_COLORS,
            dodge=False,
            legend=False,
            ax=ax,
        )
        set_axes_title(
            ax,
            f"{layer_label('dwarfs')}: распределение признака «{feature_label(feature)}» по классам",
        )
        set_axes_xlabel(ax, "Спектральный класс")
        set_axes_ylabel(ax, feature_label(feature))
        style_axes(ax)
        save_plot(f"dwarfs_{feature}_boxplot.png", figure)


def plot_scatter_layers(
    df_dwarfs_part: pd.DataFrame,
    df_evolved_part: pd.DataFrame,
) -> None:
    """Построить scatter-график карликов и evolved на плоскости Teff-radius."""
    figure, ax = make_figure_ax(FIGSIZE_STANDARD)
    draw_axes_scatter(
        ax,
        df_dwarfs_part["teff_gspphot"],
        df_dwarfs_part["radius_gspphot"],
        s=12,
        alpha=0.45,
        label=layer_label("dwarfs"),
    )
    draw_axes_scatter(
        ax,
        df_evolved_part["teff_gspphot"],
        df_evolved_part["radius_gspphot"],
        s=12,
        alpha=0.45,
        label=layer_label("evolved"),
    )
    set_axes_title(ax, "Карлики и эволюционировавшие: температура и радиус")
    set_axes_xlabel(ax, feature_label("teff_gspphot"))
    set_axes_ylabel(ax, feature_label("radius_gspphot"))
    draw_axes_legend(ax)
    style_axes(ax)
    save_plot("layers_teff_vs_radius_scatter.png", figure)


def plot_logg_radius_with_threshold(df_part: pd.DataFrame) -> None:
    """Построить scatter `logg-radius` с отмеченным dwarf threshold."""
    figure, ax = make_figure_ax(FIGSIZE_STANDARD)
    draw_scatterplot(
        data=df_part,
        x="logg_gspphot",
        y="radius_gspphot",
        hue="spec_class",
        hue_order=CLASS_ORDER,
        palette=SPEC_CLASS_COLORS,
        alpha=0.55,
        s=18,
        ax=ax,
    )
    draw_axes_vline(
        ax,
        x=LOGG_DWARF_MIN,
        color="red",
        linestyle="--",
        linewidth=1.4,
        label=f"Порог карликов: log g = {LOGG_DWARF_MIN}",
    )
    set_axes_title(ax, "Все MKGF: поверхностная гравитация и радиус")
    set_axes_xlabel(ax, feature_label("logg_gspphot"))
    set_axes_ylabel(ax, feature_label("radius_gspphot"))
    draw_axes_legend(ax)
    style_axes(ax)
    save_plot("all_logg_vs_radius_with_threshold.png", figure)


def plot_correlation_heatmaps(
    df_all: pd.DataFrame,
    df_dwarfs_part: pd.DataFrame,
    df_evolved_part: pd.DataFrame,
) -> None:
    """Построить correlation heatmap для слоёв ALL / DWARFS / EVOLVED."""
    figure, axes = make_figure_axes(3, FIGSIZE_HEATMAP_TRIPTYCH)
    layers: list[tuple[str, pd.DataFrame]] = [
        ("ALL", df_all),
        ("DWARFS", df_dwarfs_part),
        ("EVOLVED", df_evolved_part),
    ]
    for idx, (name, layer_df) in enumerate(layers):
        corr: pd.DataFrame = feature_frame(layer_df).corr()
        ax = axes[idx]
        draw_heatmap(
            data=corr,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            square=True,
            cbar=idx == 2,
            ax=ax,
        )
        set_axes_title(ax, layer_label(name.lower()) if name != "ALL" else "Все объекты")
        style_axes(ax)
    save_plot("corr_heatmaps_all_dwarfs_evolved.png", figure)
