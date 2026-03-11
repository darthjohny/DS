"""Функции построения графиков для router EDA."""

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

from analysis.router_eda.constants import (
    EVOLUTION_STAGE_LABELS,
    EVOLUTION_STAGES,
    FEATURE_LABELS,
    FEATURES,
    ROUTER_LAYER_LABELS,
    SPEC_CLASSES,
)
from analysis.router_eda.data import feature_frame
from analysis.router_eda.exports import save_plot
from analysis.visual_theme import (
    EVOLUTION_STAGE_COLORS,
    FIGSIZE_BOX,
    FIGSIZE_COUNT,
    FIGSIZE_HEATMAP_TRIPTYCH,
    FIGSIZE_STANDARD,
    FIGSIZE_WIDE,
    SPEC_CLASS_COLORS,
    set_project_theme,
    style_axes,
)


def set_default_theme() -> None:
    """Применить базовую визуальную тему для router EDA-графиков."""
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


def feature_label(feature: str) -> str:
    """Вернуть локализованное отображаемое имя признака."""
    return FEATURE_LABELS.get(feature, feature)


def relabel_legend(ax: Axes, *, title: str | None = None) -> None:
    """Заменить подписи легенды на локализованные варианты, если возможно."""
    ax_any: Any = ax
    legend = ax_any.get_legend()
    if legend is None:
        return
    handles, labels = ax_any.get_legend_handles_labels()
    localized = [EVOLUTION_STAGE_LABELS.get(label, label) for label in labels]
    ax_any.legend(handles, localized, title=title)


def localize_router_scatter_legend(ax: Axes) -> None:
    """Локализовать комбинированную legend для `spec_class + evolution_stage`."""
    ax_any: Any = ax
    legend = ax_any.get_legend()
    if legend is None:
        return
    replacements = {
        "spec_class": "Спектральный класс",
        "evolution_stage": "Эволюционная стадия",
        "dwarf": EVOLUTION_STAGE_LABELS["dwarf"],
        "evolved": EVOLUTION_STAGE_LABELS["evolved"],
    }
    legend.set_title("Обозначения")
    for text in legend.get_texts():
        label = text.get_text()
        text.set_text(replacements.get(label, label))


def plot_router_label_counts(df_router: pd.DataFrame) -> None:
    """Построить график числа объектов по `spec_class` и `evolution_stage`."""
    figure, ax = make_figure_ax(FIGSIZE_COUNT)
    draw_countplot(
        data=df_router,
        x="spec_class",
        hue="evolution_stage",
        order=SPEC_CLASSES,
        hue_order=EVOLUTION_STAGES,
        palette=EVOLUTION_STAGE_COLORS,
        ax=ax,
    )
    set_axes_title(
        ax,
        "Референсный слой router: число объектов по классам и стадиям",
    )
    set_axes_xlabel(ax, "Спектральный класс")
    set_axes_ylabel(ax, "Число объектов")
    relabel_legend(ax, title="Эволюционная стадия")
    style_axes(ax)
    save_plot("router_class_stage_counts.png", figure)


def plot_router_feature_histograms(df_router: pd.DataFrame) -> None:
    """Построить гистограммы признаков с разбиением по эволюционной стадии."""
    for feature in FEATURES:
        figure, ax = make_figure_ax(FIGSIZE_STANDARD)
        draw_histplot(
            data=df_router,
            x=feature,
            hue="evolution_stage",
            hue_order=EVOLUTION_STAGES,
            palette=EVOLUTION_STAGE_COLORS,
            bins=35,
            stat="density",
            common_norm=False,
            alpha=0.35,
            kde=True,
            ax=ax,
        )
        set_axes_title(
            ax,
            f"Референсный слой router: распределение признака «{feature_label(feature)}»",
        )
        set_axes_xlabel(ax, feature_label(feature))
        set_axes_ylabel(ax, "Плотность")
        relabel_legend(ax, title="Эволюционная стадия")
        style_axes(ax)
        save_plot(f"router_{feature}_hist.png", figure)


def plot_router_feature_boxplots(df_router: pd.DataFrame) -> None:
    """Построить boxplot-графики признаков по классам и стадиям."""
    for feature in FEATURES:
        figure, ax = make_figure_ax(FIGSIZE_BOX)
        draw_boxplot(
            data=df_router,
            x="spec_class",
            y=feature,
            hue="evolution_stage",
            order=SPEC_CLASSES,
            hue_order=EVOLUTION_STAGES,
            palette=EVOLUTION_STAGE_COLORS,
            ax=ax,
        )
        set_axes_title(
            ax,
            f"Референсный слой router: распределение признака «{feature_label(feature)}»",
        )
        set_axes_xlabel(ax, "Спектральный класс")
        set_axes_ylabel(ax, feature_label(feature))
        relabel_legend(ax, title="Эволюционная стадия")
        style_axes(ax)
        save_plot(f"router_{feature}_boxplot.png", figure)


def plot_router_scatter(
    df_router: pd.DataFrame,
    *,
    x: str,
    y: str,
    filename: str,
    title: str,
) -> None:
    """Построить один scatter-график для router reference dataset."""
    figure, ax = make_figure_ax(FIGSIZE_WIDE)
    draw_scatterplot(
        data=df_router,
        x=x,
        y=y,
        hue="spec_class",
        style="evolution_stage",
        hue_order=SPEC_CLASSES,
        style_order=EVOLUTION_STAGES,
        palette=SPEC_CLASS_COLORS,
        alpha=0.55,
        s=26,
        ax=ax,
    )
    set_axes_title(ax, title)
    set_axes_xlabel(ax, feature_label(x))
    set_axes_ylabel(ax, feature_label(y))
    localize_router_scatter_legend(ax)
    style_axes(ax)
    save_plot(filename, figure)


def plot_router_correlation_heatmaps(df_router: pd.DataFrame) -> None:
    """Построить correlation heatmap для слоёв ALL / DWARFS / EVOLVED."""
    figure, axes = make_figure_axes(3, FIGSIZE_HEATMAP_TRIPTYCH)
    layers: list[tuple[str, pd.DataFrame]] = [
        ("ALL", df_router),
        (
            "DWARFS",
            df_router[df_router["evolution_stage"] == "dwarf"].copy(),
        ),
        (
            "EVOLVED",
            df_router[df_router["evolution_stage"] == "evolved"].copy(),
        ),
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
        set_axes_title(ax, ROUTER_LAYER_LABELS.get(name, name))
        style_axes(ax)

    save_plot("router_corr_heatmaps_all_dwarfs_evolved.png", figure)
