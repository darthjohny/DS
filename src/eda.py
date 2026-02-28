# eda.py
# ============================================================
# Назначение
# ------------------------------------------------------------
# EDA для проекта ВКР (Gaia DR3 + NASA hosts).
# Цель файла:
# проверить данные
# и подготовить статистику
# для ML-этапа (Gaussian / Mahalanobis similarity).
#
# Важно: здесь разделяем 4 слоя данных:
# (A) ALL MKGF  : все объекты M/K/G/F (включая evolved)
# (B) DWARFS    : главная последовательность
#                 (logg >= 4.0)
# (C) EVOLVED   : субгиганты/гиганты (logg < 4.0)
# (D) A/B/O REF : референс-популяция A/B/O (не хосты)
# ============================================================

import os                                       # Ипорт пути для работы с файловой системой (создание папок, сохранение графиков)
from typing import Any

import matplotlib.pyplot as plt             # Библиотека для создания графиков и визуализаций
import numpy as np                          # Библиотека для численных операций (расчет mu, cov, det, eigenvalues, cond)  
import numpy.typing as npt
import pandas as pd                         # Библиотека для работы с данными в виде DataFrame (загрузка данных из SQL, обработка данных)
import seaborn as sns                       # Библиотека для статистической визуализации (гистограммы, boxplots, heatmaps)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sqlalchemy import create_engine        # Библиотека для подключения к базе данных и выполнения SQL-запросов (загрузка данных для EDA)
from sqlalchemy.engine import Engine
# ============================================================
# 1. ПАРАМЕТРЫ ПОДКЛЮЧЕНИЯ К БД
# ============================================================

FloatArray = npt.NDArray[np.floating[Any]]

USER: str = "postgres"                           # имя пользователя для подключения к Postgres (как в DBeaver)
PASSWORD: str = "1234"                           # пароль как в DBeaver
HOST: str = "127.0.0.1"                         # адрес сервера базы данных (локальный)
PORT: int = 5432                                 # порт для подключения к Postgres
DB: str = "dspro_vkr_research"                   # имя базы данных, как в DBeaver hostedatabase

engine: Engine = create_engine(                         
    f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"
)                                               # Создание SQLAlchemy engine для подключения к базе данных и выполнения SQL-запросов

sns.set_theme(style="whitegrid")                # Настройка темы для графиков seaborn (стиль с белой сеткой)

CLASS_ORDER: list[str] = ["M", "K", "G", "F"]              # Порядок классов для визуализаций (M, K, G, F)
FEATURES: list[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]   # Список признаков для анализа и визуализации (температура, логарифм гравитации, радиус)
PLOTS_DIR: str = "data/eda/plots"                    # Директория для сохранения графиков EDA
LOGG_DWARF_MIN: float = 4.0                            # Порог для отделения карликов от эволюционировавших (logg >= 4.0 для карликов)

# Пороги подклассов M.
# Верхняя граница M_EARLY
# делается строгой (< 4000),
# чтобы не пересекаться с K-областью.
M_EARLY_MIN: float = 3500.0                # Нижняя граница для M_EARLY (3500K)
M_EARLY_MAX: float = 4000.0                # Верхняя граница для M_EARLY (4000K, строго меньше, чтобы не пересекаться с K-областью)
M_MID_MIN: float = 3200.0                  # Нижняя граница для M_MID (3200K)


def read_sql_frame(query: str) -> pd.DataFrame:
    """Typed wrapper around pandas.read_sql for local EDA queries."""
    return pd.read_sql(query, engine)


def feature_frame(df_part: pd.DataFrame) -> pd.DataFrame:
    """Return the core feature subset with explicit DataFrame typing."""
    return df_part[FEATURES]


def make_figure_ax(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Create a typed matplotlib figure and single axes."""
    plt_any: Any = plt
    figure, ax = plt_any.subplots(figsize=figsize)
    return figure, ax


def make_figure_axes(
    ncols: int,
    figsize: tuple[float, float],
) -> tuple[Figure, list[Axes]]:
    """Create a typed matplotlib figure and a flat list of axes."""
    plt_any: Any = plt
    figure, axes = plt_any.subplots(1, ncols, figsize=figsize)
    axes_seq = np.atleast_1d(axes)
    typed_axes: list[Axes] = []
    for axis in axes_seq:
        typed_axes.append(axis)
    return figure, typed_axes


def draw_countplot(**kwargs: Any) -> None:
    """Typed adapter for seaborn.countplot."""
    sns_any: Any = sns
    sns_any.countplot(**kwargs)


def draw_histplot(**kwargs: Any) -> None:
    """Typed adapter for seaborn.histplot."""
    sns_any: Any = sns
    sns_any.histplot(**kwargs)


def draw_boxplot(**kwargs: Any) -> None:
    """Typed adapter for seaborn.boxplot."""
    sns_any: Any = sns
    sns_any.boxplot(**kwargs)


def draw_scatterplot(**kwargs: Any) -> None:
    """Typed adapter for seaborn.scatterplot."""
    sns_any: Any = sns
    sns_any.scatterplot(**kwargs)


def draw_heatmap(**kwargs: Any) -> None:
    """Typed adapter for seaborn.heatmap."""
    sns_any: Any = sns
    sns_any.heatmap(**kwargs)


def set_axes_title(ax: Axes, title: str) -> None:
    """Typed adapter for Axes.set_title."""
    ax_any: Any = ax
    ax_any.set_title(title)


def set_axes_xlabel(ax: Axes, label: str) -> None:
    """Typed adapter for Axes.set_xlabel."""
    ax_any: Any = ax
    ax_any.set_xlabel(label)


def set_axes_ylabel(ax: Axes, label: str) -> None:
    """Typed adapter for Axes.set_ylabel."""
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
    """Typed adapter for Axes.scatter."""
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
    """Typed adapter for Axes.axvline."""
    ax_any: Any = ax
    ax_any.axvline(
        x=x,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )


def draw_axes_legend(ax: Axes) -> None:
    """Typed adapter for Axes.legend."""
    ax_any: Any = ax
    ax_any.legend()


def save_plot(filename: str, figure: Figure) -> None:               # Функция для сохранения текущего графика matplotlib в указанную директорию с заданным именем файла                          
    """Сохраняет текущий matplotlib-график."""                                                      
    os.makedirs(PLOTS_DIR, exist_ok=True)           # Создание директории для графиков, если она не существует                                     
    path = os.path.join(PLOTS_DIR, filename)        # Полный путь для сохранения графика
    figure_any: Any = figure
    plt_any: Any = plt
    figure_any.tight_layout()                       # Улучшение компоновки графика перед сохранением (убирает лишние отступы)
    figure_any.savefig(path, dpi=160)               # Сохранение графика в файл с разрешением 160 dpi
    plt_any.close(figure)                           # Закрытие текущей фигуры, чтобы освободить память и избежать наложения графиков при следующем вызове save_plot 


def plot_class_counts(          # Функция для построения графика количества объектов по классам (M/K/G/F) для заданного DataFrame, заголовка и имени файла для сохранения графика
    df_part: pd.DataFrame,      # Часть DataFrame для анализа (например, ALL MKGF, DWARFS или EVOLVED)
    title: str,                 # Заголовок для графика (например, "ALL MKGF: число объектов по классам")
    filename: str,              # Имя файла для сохранения графика (например, "all_class_counts.png")
) -> None:                     
    """График количества объектов
    по классам.
    """
    figure, ax = make_figure_ax((7, 4))            # Создание новой фигуры и оси для графика с размером 7x4 дюйма
    draw_countplot(              # Построение графика количества объектов по классам с помощью seaborn countplot
        data=df_part,           #Данные для графика (часть DataFrame)
        x="spec_class",         # Ось X - спектральный класс (M/K/G/F)                           
        hue="spec_class",       # Цвета для классов (hue) - тот же столбец "spec_class" для стабильного поведения seaborn при сохранении порядка классов
        order=CLASS_ORDER,      #Порядок классов на оси X (M, K, G, F)
        hue_order=CLASS_ORDER,  #Порядок классов для цветов (hue) - тот же порядок, что и для оси X
        palette="viridis",      # Цветовая палитра для классов (viridis)
        legend=False,           # Отключение легенды, так как цвет уже соответствует классу на оси X
        ax=ax,                  # Указание оси для построения графика (ax)
    )
    set_axes_title(ax, title)          # Установка заголовка для графика
    set_axes_xlabel(ax, "spec_class")  #Установка подписи для оси X
    set_axes_ylabel(ax, "count")       # Установка подписи для оси Y
    save_plot(filename, figure)          # Сохранение графика с помощью функции save_plot, передавая имя файла для сохранения (например, "all_class_counts.png")


def plot_feature_histograms(    # Функция для построения гистограмм распределения признаков (teff_gspphot, logg_gspphot, radius_gspphot) по классам для заданного DataFrame и префикса для заголовков и имен файлов
    df_part: pd.DataFrame,      # Часть DataFrame для анализа (например, ALL MKGF, DWARFS или EVOLVED)
    prefix: str,                # Префикс для заголовков и имен файлов (например, "all" для ALL MKGF, "dwarfs" для DWARFS, "evolved" для EVOLVED)
) -> None:
    """Гистограммы признаков по классам."""
    for feature in FEATURES:    # Проход по каждому признаку (teff_gspphot, logg_gspphot, radius_gspphot) для построения гистограмм
        figure, ax = make_figure_ax((8, 5))           # Создание новой фигуры и оси для графика с размером 8x5 дюйма
        draw_histplot(           # Построение гистограммы распределения признака по классам с помощью seaborn histplot
            data=df_part,       # Данные для графика (часть DataFrame)
            x=feature,          # Ось X - текущий признак (teff_gspphot, logg_gspphot или radius_gspphot)
            hue="spec_class",   # Цвета для классов (hue) - столбец "spec_class" для раздел
            hue_order=CLASS_ORDER,      # Порядок классов для цветов (hue) - тот же порядок, что и для оси X
            bins=30,               # Количество бинов для гистограммы (30)
            stat="density",        # Нормировка гистограммы по плотности (density), чтобы сравнивать распределения разных классов
            common_norm=False,     # Отключение общей нормировки, чтобы каждый класс отображался по своей плотности, а не по общему количеству объектов
            alpha=0.35,            # Прозрачность для гистограмм классов (0.35), чтобы видеть наложение распределений
            kde=True,              # Добавление KDE (ядерной оценки плотности) для сглаживания распределения каждого класса
            ax=ax,                 # Указание оси для построения графика (ax)
        )
        set_axes_title(ax, f"{prefix}: распределение {feature}")  # Установка заголовка для графика, включающего префикс (например, "all", "dwarfs" или "evolved") и название признака (teff_gspphot, logg_gspphot или radius_gspphot)
        set_axes_xlabel(ax, feature)                 # Установка подписи для оси X, которая соответствует названию текущего признака
        set_axes_ylabel(ax, "density")        # Установка подписи для оси Y, которая соответствует плотности распределения (density)
        save_plot(f"{prefix}_{feature}_hist.png", figure)   # Сохранение графика с помощью функции save_plot, передавая имя файла для сохранения, которое включает префикс и название признака (например, "all_teff_gspphot_hist.png", "dwarfs_logg_gspphot_hist.png" и т.д.)


def plot_boxplots_dwarfs(df_part: pd.DataFrame) -> None:     # Функция для построения boxplot'ов признаков (teff_gspphot, logg_gspphot, radius_gspphot) по классам для слоя карликов (DWARFS)
    """Boxplot признаков для слоя карликов."""
    for feature in FEATURES:        # Проход по каждому признаку (teff_gspphot, logg_gspphot, radius_gspphot) для построения boxplot'ов
        figure, ax = make_figure_ax((8, 5)) # Создание новой фигуры и оси для графика с размером 8x5 дюйма
        draw_boxplot(                 # Построение boxplot'ов для признака по классам с помощью seaborn boxplot
            data=df_part,            # Данные для графика (часть DataFrame, которая соответствует слою карликов - DWARFS)
            x="spec_class",          # Ось X - спектральный класс (M/K/G/F) для boxplot'ов
            y=feature,               # Ось Y - текущий признак (teff_gspphot, logg_gspphot или radius_gspphot) для boxplot'ов
            hue="spec_class",        # Цвета для классов (hue) - тот же столбец "spec_class" для стабильного поведения seaborn при сохранении порядка классов
            order=CLASS_ORDER,       # Порядок классов на оси X (M, K, G, F)
            hue_order=CLASS_ORDER,   # Порядок классов для цветов (hue) - тот же порядок, что и для оси X
            palette="Set2",          # Цветовая палитра для классов (Set2)
            dodge=False,             # Отключение раздвигания boxplot'ов по классам (dodge=False), чтобы все классы были на одной позиции по оси X и не было наложения boxplot'ов для разных классов
            legend=False,            # Отключение легенды, так как цвет уже соответствует классу на оси X
            ax=ax,                   # Указание оси для построения графика (ax)  
        )
        set_axes_title(ax, f"DWARFS: boxplot {feature}")  # Установка заголовка для графика, включающего название слоя (DWARFS) и название признака (teff_gspphot, logg_gspphot или radius_gspphot)
        set_axes_xlabel(ax, "spec_class")         # Установка подписи для оси X, которая соответствует спектральному классу (M/K/G/F)
        set_axes_ylabel(ax, feature)              #Установка подписи для оси Y, которая соответствует названию текущего признака (teff_gspphot, logg_gspphot или radius_gspphot)
        save_plot(f"dwarfs_{feature}_boxplot.png", figure) # Сохранение графика с помощью функции save_plot, передавая имя файла для сохранения, которое включает название слоя (dwarfs) и название признака (например, "dwarfs_teff_gspphot_boxplot.png", "dwarfs_logg_gspphot_boxplot.png" и т.д.)


def plot_scatter_layers(                # Функция для построения scatter-графика сравнения слоев DWARFS и EVOLVED по признакам teff_gspphot и radius_gspphot
    df_dwarfs_part: pd.DataFrame,       # Часть DataFrame для слоя карликов (DWARFS)
    df_evolved_part: pd.DataFrame,      # Часть DataFrame для слоя эволюционировавших (EVOLVED)
) -> None:
    """Scatter для сравнения слоёв DWARFS и EVOLVED."""
    figure, ax = make_figure_ax((8, 5))    # Создание новой фигуры и оси для графика с размером 8x5 дюйма
    draw_axes_scatter(                      # Построение scatter-графика для слоя карликов (DWARFS) по признакам teff_gspphot и radius_gspphot
        ax,
        df_dwarfs_part["teff_gspphot"],     #Ось X - температура для слоя карликов (DWARFS)
        df_dwarfs_part["radius_gspphot"],   # Ось Y - радиус для слоя карликов (DWARFS)
        s=12,                               # Размер точек для слоя карликов (DWARFS)
        alpha=0.45,                         # Прозрачность точек для слоя карликов (DWARFS)
        label="DWARFS",                     # Метка для слоя карликов (DWARFS) в легенде графика
    )
    draw_axes_scatter(                      # Построение scatter-графика для слоя эволюционировавших (EVOLVED) по признакам teff_gspphot и radius_gspphot
        ax,
        df_evolved_part["teff_gspphot"],    # Ось X - температура для слоя эволюционировавших (EVOLVED)
        df_evolved_part["radius_gspphot"],  # Ось Y - радиус для слоя эволюционировавших (EVOLVED)
        s=12,                               # Размер точек для слоя эволюционировавших (EVOLVED)
        alpha=0.45,                         # Прозрачность точек для слоя эволюционировавших (EVOLVED)
        label="EVOLVED",                    # Метка для слоя эволюционировавших (EVOLVED) в легенде графика
    )
    set_axes_title(ax, "Teff vs Radius: DWARFS и EVOLVED")    # Установка заголовка для графика, который описывает сравнение слоев DWARFS и EVOLVED по признакам teff_gspphot и radius_gspphot
    set_axes_xlabel(ax, "teff_gspphot")               # Установка подписи для оси X, которая соответствует температуре (teff_gspphot)
    set_axes_ylabel(ax, "radius_gspphot")             # Установка подписи для оси Y, которая соответствует радиусу (radius_gspphot)
    draw_axes_legend(ax)                               # Установка легенды для графика, которая отображает метки для слоев DWARFS и EVOLVED
    save_plot("layers_teff_vs_radius_scatter.png", figure)  # Сохранение графика с помощью функции save_plot, передавая имя файла для сохранения (например, "layers_teff_vs_radius_scatter.png")


def plot_logg_radius_with_threshold(df_part: pd.DataFrame) -> None:     # Функция для построения scatter-графика logg_gspphot vs radius_gspphot для всех MKGF с выделением порога logg = 4.0, который отделяет карликов от эволюционировавших
    """Scatter logg-radius и порог карликов logg = 4.0."""     # Этот график помогает визуально оценить, как объекты распределяются по признакам logg_gspphot и radius_gspphot, и насколько четко порог logg = 4.0 отделяет карликов от эволюционировавших. Если объекты с logg >= 4.0 (карлики) сгруппированы в одной области графика, а объекты с logg < 4.0 (эволюционировавшие) - в другой, это подтверждает правильность выбора порога для разделения слоев данных.
    figure, ax = make_figure_ax((8, 5))    # Создание новой фигуры и оси для графика с размером 8x5 дюйма
    draw_scatterplot(                        # Построение scatter-графика logg_gspphot vs radius_gspphot для всех MKGF с помощью seaborn scatterplot
        data=df_part,                       # Данные для графика (часть DataFrame, которая соответствует всем MKGF)
        x="logg_gspphot",                   # Ось X - логарифм гравитации (logg_gspphot) для всех MKGF
        y="radius_gspphot",                 # Ось Y - радиус (radius_gspphot) для всех MKGF
        hue="spec_class",                   # Цвета для классов (hue) - столбец "spec_class" для раздел классов (M/K/G/F)
        hue_order=CLASS_ORDER,              # Порядок классов для цветов (hue) - тот же порядок, что и для оси X
        alpha=0.55,                         # Прозрачность точек для всех MKGF (0.55), чтобы видеть наложение объектов и общую структуру распределения по признакам logg_gspphot и radius_gspphot
        s=18,                               # Размер точек для всех MKGF (18), чтобы объекты были хорошо видны на графике
        ax=ax,                              # Указание оси для построения графика (ax)
    )
    draw_axes_vline(                        # Добавление вертикальной линии на график, которая соответствует порогу logg = 4.0, отделяющему карликов от эволюционировавших
        ax,
        x=LOGG_DWARF_MIN,                   # Порог для отделения карликов от эволюционировавших (logg = 4.0)
        color="red",                        # Цвет линии для порога (красный)
        linestyle="--",                     # Стиль линии для порога (штриховая линия)
        linewidth=1.4,                      # Толщина линии для порога (1.4)
        label=f"logg = {LOGG_DWARF_MIN}",   # Метка для линии порога в легенде графика (например, "logg = 4.0")
    )
    set_axes_title(ax, "ALL MKGF: logg vs radius")    # Установка заголовка для графика, который описывает сравнение всех MKGF по признакам logg_gspphot и radius_gspphot с выделением порога logg = 4.0
    set_axes_xlabel(ax, "logg_gspphot")               # Установка подписи для оси X, которая соответствует логарифму гравитации (logg_gspphot)
    set_axes_ylabel(ax, "radius_gspphot")             # Установка подписи для оси Y, которая соответствует радиусу (radius_gspphot)
    draw_axes_legend(ax)                               # Установка легенды для графика, которая отображает метки для классов (M/K/G/F) и порога logg = 4.0
    save_plot("all_logg_vs_radius_with_threshold.png", figure) # Сохранение графика с помощью функции save_plot, передавая имя файла для сохранения (например, "all_logg_vs_radius_with_threshold.png")


def plot_correlation_heatmaps(                  # Функция для построения тепловых карт корреляций между признаками teff_gspphot, logg_gspphot и radius_gspphot для всех MKGF, слоя карликов (DWARFS) и слоя эволюционировавших (EVOLVED)
    df_all: pd.DataFrame,                       # Часть DataFrame для всех MKGF (ALL MKGF)
    df_dwarfs_part: pd.DataFrame,               # Часть DataFrame для слоя карликов (DWARFS)
    df_evolved_part: pd.DataFrame,              # Часть DataFrame для слоя эволюционировавших (EVOLVED)
) -> None:
    """Heatmap корреляций для ALL / DWARFS / EVOLVED."""
    figure, axes = make_figure_axes(3, (15, 4))       # Создание новой фигуры и массива осей для графиков с размером 15x4 дюйма и 1 строкой, 3 столбцами (для ALL MKGF, DWARFS и EVOLVED)
    layers: list[tuple[str, pd.DataFrame]] = [          # Список слоев данных для построения тепловых карт корреляций, который включает кортежи с названием слоя и соответствующей частью DataFrame
        ("ALL", df_all),                        # Слой для всех MKGF (ALL MKGF) с соответствующей частью DataFrame (df_all)
        ("DWARFS", df_dwarfs_part),             # Слой для карликов (DWARFS) с соответствующей частью DataFrame (df_dwarfs_part)
        ("EVOLVED", df_evolved_part),           # Слой для эволюционировавших (EVOLVED) с соответствующей частью DataFrame (df_evolved_part)
    ]
    for idx, (name, layer_df) in enumerate(layers):     # Проход по каждому слою данных (ALL MKGF, DWARFS, EVOLVED) с помощью enumerate для получения индекса и кортежа с названием слоя и соответствующей частью DataFrame
        corr: pd.DataFrame = feature_frame(layer_df).corr()    # Вычисление матрицы корреляций между признаками teff_gspphot, logg_gspphot и radius_gspphot для текущего слоя данных (layer_df) с помощью метода corr() для DataFrame, который возвращает матрицу корреляций
        ax = axes[idx]
        draw_heatmap(                # Построение тепловой карты корреляций для текущего слоя данных (layer_df) с помощью seaborn heatmap
            data=corr,              # Матрица корреляций между признаками для текущего слоя данных (layer_df)
            vmin=-1.0,              # Минимальное значение для цветовой шкалы (vmin=-1.0), чтобы отображать отрицательные корреляции красным цветом
            vmax=1.0,               # Максимальное значение для цветовой шкалы (vmax=1.0), чтобы отображать положительные корреляции синим цветом
            center=0.0,             # Центр цветовой шкалы (center=0.0), чтобы нейтральные корреляции (около 0) отображались белым цветом
            cmap="coolwarm",        #Цветовая палитра для тепловой карты (coolwarm), которая отображает отрицательные корреляции красным цветом, положительные - синим цветом, а нейтральные - белым цветом
            annot=True,             # Включение отображения числовых значений корреляций на тепловой карте (annot=True)
            fmt=".2f",              # Формат отображения числовых значений корреляций с двумя десятичными знаками (fmt=".2f")
            square=True,            # Отображение тепловой карты в виде квадратов (square=True) для лучшей визуализации корреляций между признаками
            cbar=idx == 2,          # Отображение цветовой шкалы только для последнего графика (EVOLVED) для экономии места и улучшения визуального восприятия (cbar=idx == 2)
            ax=ax,           # Указание соответствующей оси для построения тепловой карты корреляций для текущего слоя данных (layer_df) (ax=axes[idx] - выбор оси по индексу)
        )
        set_axes_title(ax, name)   # Установка заголовка для тепловой карты корреляций, который соответствует названию слоя данных (name) (например, "ALL", "DWARFS" или "EVOLVED")
    save_plot("corr_heatmaps_all_dwarfs_evolved.png", figure)   # Сохранение графика с помощью функции save_plot, передавая имя файла для сохранения (например, "corr_heatmaps_all_dwarfs_evolved.png")


# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ ДЛЯ EDA
# ============================================================

# Источники:
# - lab.v_nasa_gaia_train_classified : MKGF хосты с spec_class
# - lab.v_nasa_gaia_train_dwarfs     : MKGF, только logg >= 4
# - lab.v_nasa_gaia_train_evolved    : MKGF, только logg < 4

# Базовая обучающая выборка
# (все MKGF, включая evolved)
QUERY_ALL_MKGF: str = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M','K','G','F');
"""

# Карлики для обучения гауссовской модели
QUERY_DWARFS_MKGF: str = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_dwarfs
WHERE spec_class IN ('M','K','G','F');
"""

# Эволюционировавшие
# (отдельный аналитический слой)
QUERY_EVOLVED_MKGF: str = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_evolved
WHERE spec_class IN ('M','K','G','F');
"""

# Загружаем данные
# ALL MKGF: общая картина
# DWARFS: основа для mu/cov
# EVOLVED: отдельный слой

df: pd.DataFrame = read_sql_frame(QUERY_ALL_MKGF)                # Загрузка данных для всех MKGF (включая evolved) из базы данных с помощью SQL-запроса QUERY_ALL_MKGF и сохранение результата в DataFrame df
df_dwarfs: pd.DataFrame = read_sql_frame(QUERY_DWARFS_MKGF)      # Загрузка данных для слоя карликов (DWARFS) из базы данных с помощью SQL-запроса QUERY_DWARFS_MKGF и сохранение результата в DataFrame df_dwarfs
df_evolved: pd.DataFrame = read_sql_frame(QUERY_EVOLVED_MKGF)    # Загрузка данных для слоя эволюционировавших звезд (EVOLVED) из базы данных с помощью SQL-запроса QUERY_EVOLVED_MKGF и сохранение результата в DataFrame df_evolved

print("\n=== ДАННЫЕ ЗАГРУЖЕНЫ ===")                     # Вывод сообщения о том, что данные успешно загружены из базы данных для всех MKGF, слоя карликов (DWARFS) и слоя эволюционировавших звезд (EVOLVED)
print("Размер ALL MKGF:", df.shape)
print("Размер DWARFS (logg>=4.0):", df_dwarfs.shape)    # Вывод размера выборки для слоя карликов (DWARFS), который включает объекты с logg >= 4.0
print("Размер EVOLVED (logg<4.0):", df_evolved.shape)   # Вывод размера выборки для слоя эволюционировавших звезд (EVOLVED), который включает объекты с logg < 4.0


# ============================================================
# 3. ПРОВЕРКА NULL (ALL MKGF)
# ============================================================

print("\n=== ПРОВЕРКА NULL ===")                         # Вывод заголовка для раздела проверки наличия пропущенных значений (NULL) в данных для всех MKGF
print("NULL по столбцам (ALL MKGF):\n", df.isnull().sum())  # Вывод количества пропущенных значений (NULL) по каждому столбцу для всех MKGF с помощью метода isnull().sum() для DataFrame df, который возвращает количество пропущенных значений для каждого столбца (spec_class, teff_gspphot, logg_gspphot, radius_gspphot)


# ============================================================
# 4. ОБЩИЕ СТАТИСТИКИ (ALL MKGF)
# ============================================================

print("\n=== ОБЩАЯ СТАТИСТИКА (describe) ===")           # Вывод заголовка для раздела общей статистики данных для всех MKGF с помощью метода describe(), который предоставляет основные статистические показатели (count, mean, std, min, 25%, 50%, 75%, max) для числовых столбцов (teff_gspphot, logg_gspphot, radius_gspphot) в DataFrame df
print(df.describe())                                     # Вывод общей статистики для всех MKGF с помощью метода describe() для DataFrame df, который включает числовые столбцы teff_gspphot, logg_gspphot и radius_gspphot, и предоставляет основные статистические показатели для этих признаков (количество объектов, среднее значение, стандартное отклонение, минимальное значение, 25-й перцентиль, медиану (50-й перцентиль), 75-й перцентиль и максимальное значение)


# ============================================================
# 5. КОРРЕЛЯЦИИ (ALL MKGF)
# ============================================================

print("\n=== КОРРЕЛЯЦИИ МЕЖДУ ПРИЗНАКАМИ ===")             # Вывод заголовка для раздела корреляций между признаками для всех MKGF, который поможет понять взаимосвязи между признаками teff_gspphot, logg_gspphot и radius_gspphot в данных для всех MKGF
print(df[["teff_gspphot", "logg_gspphot", "radius_gspphot"]].corr())  # Вывод матрицы корреляций между признаками teff_gspphot, logg_gspphot и radius_gspphot для всех MKGF с помощью метода corr() для DataFrame df, который возвращает матрицу корреляций, показывающую степень линейной взаимосвязи между этими признаками (значения от -1 до 1, где 1 означает сильную положительную корреляцию, -1 - сильную отрицательную корреляцию, а 0 - отсутствие линейной корреляции)


# ============================================================
# 6. СТАТИСТИКА ПО КЛАССАМ (ALL MKGF: M/K/G/F)
# ============================================================

print(                                                      # Вывод заголовка для раздела статистики по спектральным классам для всех MKGF, который поможет понять основные статистические показатели (среднее, стандартное отклонение, минимальное и максимальное значение) для признаков teff_gspphot, logg_gspphot и radius_gspphot в разрезе каждого спектрального класса (M/K/G/F) в данных для всех MKGF
    "\n=== СТАТИСТИКА "
    "ПО СПЕКТРАЛЬНЫМ КЛАССАМ ==="
)

group_stats: pd.DataFrame = (                                             # Вычисление статистических показателей (среднее, стандартное отклонение, минимальное и максимальное значение) для признаков teff_gspphot, logg_gspphot и radius_gspphot в разрезе каждого спектрального класса (M/K/G/F) для всех MKGF с помощью метода groupby() для DataFrame df, который группирует данные по столбцу "spec_class", и метода agg(), который применяет функции агрегации (mean, std, min, max) к выбранным признакам для каждой группы.
    df.groupby("spec_class")[       
        ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
    ].agg(["mean", "std", "min", "max"])
)

print(group_stats)                                          # Вывод статистических показателей для каждого спектрального класса (M/K/G/F) в данных для всех MKGF, который включает среднее значение, стандартное отклонение, минимальное и максимальное значение для признаков teff_gspphot, logg_gspphot и radius_gspphot в разрезе каждого класса.


# ============================================================
# 7. БЫСТРАЯ ПРОВЕРКА EVOLVED НА ALL MKGF
# ============================================================

print(                                                      # Вывод заголовка для раздела быстрой проверки наличия эволюционировавших звезд (EVOLVED) в данных для всех MKGF по признаку радиуса (radius_gspphot), который поможет визуально оценить, есть ли объекты с большими радиусами, которые могут соответствовать эволюционировавшим звездам, и насколько они выражены в данных для всех MKGF
    "\n=== ALL MKGF: ТОП-20 ПО РАДИУСУ "
    "(быстрая проверка EVOLVED/гигантов) ==="
)

top_radius: pd.DataFrame = df.sort_values("radius_gspphot", ascending=False).head(20) # Выбор топ-20 объектов с наибольшими радиусами (radius_gspphot) из данных для всех MKGF с помощью метода sort_values() для DataFrame df, который сортирует данные по столбцу "radius_gspphot" в порядке убывания (ascending=False), и метода head(20), который выбирает первые 20 строк из отсортированных данных. Этот выбор поможет быстро проверить наличие объектов с большими радиусами, которые могут соответствовать эволюционировавшим звездам (EVOLVED) или гигантам, в данных для всех MKGF.
print(top_radius)


# ============================================================
# 8. СЛОЙ DWARFS (ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ)
# ============================================================

print(                                                  # Вывод заголовка для раздела слоя карликов (DWARFS), который является главной последовательностью звезд с logg >= 4.0, и который будет использоваться для построения многомерной гауссовой модели. Этот раздел поможет сосредоточиться на анализе и статистике объектов, которые соответствуют карликам, и исключить объекты с logg < 4.0, которые относятся к эволюционировавшим звездам (EVOLVED).
    "\n=== ТОЛЬКО ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ "
    "(из view: v_nasa_gaia_train_dwarfs) ==="
)
print("Размер выборки:", df_dwarfs.shape)               # Вывод размера выборки для слоя карликов (DWARFS), который включает объекты с logg >= 4.0, и который будет использоваться для построения многомерной гауссовой модели. Этот размер поможет понять, сколько объектов соответствует критерию карликов в данных для всех MKGF.

print("\n=== СТАТИСТИКА ПОСЛЕ ФИЛЬТРА ===")             # Вывод заголовка для раздела статистики после фильтрации данных для слоя карликов (DWARFS), который поможет понять основные статистические показатели (среднее, стандартное отклонение, минимальное и максимальное значение) для признаков teff_gspphot, logg_gspphot и radius_gspphot в данных для слоя карликов (DWARFS) с logg >= 4.0. Этот раздел поможет оценить, как изменились статистические показатели после применения фильтра по logg и сосредоточиться на анализе объектов, которые соответствуют карликам.
print(df_dwarfs.describe())

print("\n=== КОРРЕЛЯЦИИ ПОСЛЕ ФИЛЬТРА ===")             # Вывод заголовка для раздела корреляций после фильтрации данных для слоя карликов (DWARFS), который поможет понять взаимосвязи между признаками teff_gspphot, logg_gspphot и radius_gspphot в данных для слоя карликов (DWARFS) с logg >= 4.0. Этот раздел поможет оценить, как изменились корреляции между признаками после применения фильтра по logg и сосредоточиться на анализе объектов, которые соответствуют карликам.
print(df_dwarfs[["teff_gspphot", "logg_gspphot", "radius_gspphot"]].corr()) # Вывод матрицы корреляций между признаками teff_gspphot, logg_gspphot и radius_gspphot для слоя карликов (DWARFS) с logg >= 4.0 с помощью метода corr() для DataFrame df_dwarfs, который возвращает матрицу корреляций, показывающую степень линейной взаимосвязи между этими признаками в данных для слоя карликов (DWARFS)


# ============================================================
# 9. СЛОЙ EVOLVED (logg < 4.0)
# ============================================================

# Эти объекты не входят
# в гауссову модель карликов,
# но сохраняются для аналитики и отчёта.
df_anomalies: pd.DataFrame = df_evolved.copy()                 # Создание копии DataFrame df_evolved, который содержит объекты с logg < 4.0 (эволюционировавшие звезды), и сохранение его в переменную df_anomalies для дальнейшего анализа и отчетности. Эти объекты не будут включены в многомерную гауссову модель для карликов (DWARFS), но будут сохранены для аналитики и отчетности, чтобы понять характеристики эволюционировавших звезд в данных для всех MKGF.

print(                                           # Вывод заголовка для раздела слоя эволюционировавших звезд (EVOLVED) с logg < 4.0, который является отдельным аналитическим слоем и не будет включен в многомерную гауссову модель для карликов (DWARFS). Этот раздел поможет сосредоточиться на анализе и статистике объектов, которые соответствуют эволюционировавшим звездам, и понять их характеристики в данных для всех MKGF.
    "\n=== EVOLVED (logg < 4.0): "
    "отдельный аналитический слой ==="
)
print("Количество:", df_anomalies.shape)         # Вывод количества объектов в слое эволюционировавших звезд (EVOLVED) с logg < 4.0, который является отдельным аналитическим слоем и не будет включен в многомерную гауссову модель для карликов (DWARFS). Этот вывод поможет понять, сколько объектов соответствует критерию эволюционировавших звезд в данных для всех MKGF.

os.makedirs("data/eda", exist_ok=True)           # Создание директории "data/eda" для сохранения результатов EDA (Exploratory Data Analysis) и графиков, если она еще не существует, с помощью функции os.makedirs() и параметра exist_ok=True, который позволяет избежать ошибки, если директория уже существует.
df_anomalies.to_csv("data/eda/evolved_stars_snapshot.csv", index=False)     # Сохранение данных для слоя эволюционировавших звезд (EVOLVED) с logg < 4.0 в CSV-файл "evolved_stars_snapshot.csv" в директории "data/eda" с помощью метода to_csv() для DataFrame df_anomalies, который сохраняет данные в формате CSV, и параметра index=False, который исключает сохранение индекса DataFrame в файле.


# ============================================================
# 9.1. СЛОЙ A/B/O REF (не хосты)
# ============================================================

print(                                           # Вывод заголовка для раздела слоя A/B/O REF, который включает объекты спектральных классов A, B и O из вьюхи lab.v_gaia_ref_abo_training, и который не является частью хостов (MKGF), но может быть полезен для сравнения и аналитики. Этот раздел поможет сосредоточиться на анализе и статистике объектов с высокими температурами (A/B/O классы) в данных для всех MKGF, и понять их характеристики в сравнении с карликами (DWARFS) и эволюционировавшими звездами (EVOLVED).
    "\n=== A/B/O REF (не хосты): "
    "диапазоны v_gaia_ref_abo_training ==="
)

QUERY_ABO_REF: str = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_gaia_ref_abo_training
WHERE spec_class IN ('A','B','O');
"""

df_abo: pd.DataFrame = read_sql_frame(QUERY_ABO_REF)     # Загрузка данных для слоя A/B/O REF из базы данных с помощью SQL-запроса QUERY_ABO_REF и сохранение результата в DataFrame df_abo, который содержит объекты спектральных классов A, B и O, и который не является частью хостов (MKGF), но может быть полезен для сравнения и аналитики с карликами (DWARFS) и эволюционировавшими звездами (EVOLVED) в данных для всех MKGF.
print("Размер ABO ref:", df_abo.shape)          # Вывод размера выборки для слоя A/B/O REF, который включает объекты спектральных классов A, B и O, и который не является частью хостов (MKGF), но может быть полезен для сравнения и аналитики с карликами (DWARFS) и эволюционировавшими звездами (EVOLVED) в данных для всех MKGF. Этот размер поможет понять, сколько объектов соответствует критерию A/B/O классов в данных для всех MKGF.
print("\n=== ABO ref: describe ===")            # Вывод заголовка для раздела описательной статистики для слоя A/B/O REF, который поможет понять основные статистические показатели (среднее, стандартное отклонение, минимальное и максимальное значение) для признаков teff_gspphot, logg_gspphot и radius_gspphot в данных для слоя A/B/O REF, который включает объекты спектральных классов A, B и O, и который не является частью хостов (MKGF), но может быть полезен для сравнения и аналитики с карликами (DWARFS) и эволюционировавшими звездами (EVOLVED) в данных для всех MKGF.
print(df_abo.describe())                        # Вывод описательной статистики для слоя A/B/O REF, который включает объекты спектральных классов A, B и O, и который не является частью хостов (MKGF), но может быть полезен для сравнения и аналитики с карликами (DWARFS) и эволюционировавшими звездами (EVOLVED) в данных для всех MKGF. Этот вывод поможет понять основные статистические показатели для признаков teff_gspphot, logg_gspphot и radius_gspphot в данных для слоя A/B/O REF.

print("\n=== ABO ref: статистика по классам ===")   # Вывод заголовка для раздела статистики по спектральным классам для слоя A/B/O REF, который поможет понять основные статистические показатели (среднее, стандартное отклонение, минимальное и максимальное значение) для признаков teff_gspphot, logg_gspphot и radius_gspphot в разрезе каждого спектрального класса (A/B/O) в данных для слоя A/B/O REF, который включает объекты спектральных классов A, B и O, и который не является частью хостов (MKGF), но может быть полезен для сравнения и аналитики с карликами (DWARFS) и эволюционировавшими звездами (EVOLVED) в данных для всех MKGF.
abo_stats: pd.DataFrame = (                                       # Вычисление статистических показателей (количество, среднее значение, стандартное отклонение, минимальное и максимальное значение) для признаков teff_gspphot, logg_gspphot и radius_gspphot в разрезе каждого спектрального класса (A/B/O) для слоя A/B/O REF с помощью метода groupby() для DataFrame df_abo, который группирует данные по столбцу "spec_class", и метода agg(), который применяет функции агрегации (count, mean, std, min, max) к выбранным признакам для каждой группы.
    df_abo.groupby("spec_class")[FEATURES]          # Группировка данных для слоя A/B/O REF по столбцу "spec_class" (A/B/O) с помощью метода groupby() для DataFrame df_abo, который позволяет разделить данные на группы в зависимости от спектрального класса (A/B/O), и выбор признаков teff_gspphot, logg_gspphot и radius_gspphot для дальнейшего анализа.
    .agg(["count", "mean", "std", "min", "max"])    # Применение функций агрегации (count, mean, std, min, max) к выбранным признакам для каждой группы в слое A/B/O REF с помощью метода agg(), который позволяет вычислить количество объектов (count), среднее значение (mean), стандартное отклонение (std), минимальное значение (min) и максимальное значение (max) для признаков teff_gspphot, logg_gspphot и radius_gspphot в разрезе каждого спектрального класса (A/B/O) в данных для слоя A/B/O REF.
)
print(abo_stats)                                     # Вывод статистических показателей для каждого спектрального класса (A/B/O) в данных для слоя A/B/O REF, который включает объекты спектральных классов A, B и O, и который не является частью хостов (MKGF), но может быть полезен для сравнения и аналитики с карликами (DWARFS) и эволюционировавшими звездами (EVOLVED) в данных для всех MKGF. Этот вывод поможет понять основные статистические показатели для признаков teff_gspphot, logg_gspphot и radius_gspphot в разрезе каждого класса (A/B/O) в данных для слоя A/B/O REF.

df_abo.sort_values("teff_gspphot", ascending=False).head(20).to_csv(    # Выбор топ-20 объектов с наибольшими температурами (teff_gspphot) из данных для слоя A/B/O REF с помощью метода sort_values() для DataFrame df_abo, который сортирует данные по столбцу "teff_gspphot" в порядке убывания (ascending=False), и метода head(20), который выбирает первые 20 строк из отсортированных данных, и сохранение этих данных в CSV-файл "abo_top20_by_teff.csv" в директории "data/eda" с помощью метода to_csv() для DataFrame, который сохраняет данные в формате CSV, и параметра index=False, который исключает сохранение индекса DataFrame в файле. Этот выбор поможет быстро проверить наличие объектов с высокими температурами, которые соответствуют классам A/B/O, в данных для слоя A/B/O REF.
    "data/eda/abo_top20_by_teff.csv",
    index=False,                                     # Исключение сохранения индекса DataFrame в CSV-файле для топ-20 объектов с наибольшими температурами (teff_gspphot) из данных для слоя A/B/O REF, который включает объекты спектральных классов A, B и O, и который не является частью хостов (MKGF), но может быть полезен для сравнения и аналитики с карликами (DWARFS) и эволюционировавшими звездами (EVOLVED) в данных для всех MKGF. Этот выбор поможет сохранить только данные без индекса в CSV-файле для дальнейшего анализа и отчетности.
)


# ============================================================
# 9.2. ВИЗУАЛИЗАЦИЯ EDA
# ============================================================

print(                                  # Вывод заголовка для раздела визуализации EDA, который включает сохранение графиков для всех MKGF, слоя карликов (DWARFS) и слоя эволюционировавших звезд (EVOLVED). Этот раздел поможет сосредоточиться на визуальном представлении данных и взаимосвязей между признаками для разных слоев данных, а также сохранить эти графики для дальнейшего анализа и отчетности.
    "\n=== ВИЗУАЛИЗАЦИЯ EDA: "
    "сохраняем графики ==="
)

plot_class_counts(                      # Построение графика количества объектов по классам для всех MKGF (ALL MKGF) с помощью функции plot_class_counts, которая принимает DataFrame
    df,
    "ALL MKGF: число объектов по классам",
    "all_class_counts.png",
)
plot_class_counts(                      # Построение графика количества объектов по классам для слоя карликов (DWARFS) с помощью функции plot_class_counts, которая принимает DataFrame df_dwarfs, заголовок для графика и имя файла для сохранения. Этот график поможет визуально оценить распределение объектов по спектральным классам (M/K/G/F) в слое карликов (DWARFS) с logg >= 4.0, который будет использоваться для построения многомерной гауссовой модели.
    df_dwarfs,
    "DWARFS: число объектов по классам",
    "dwarfs_class_counts.png",
)
plot_class_counts(                      # Построение графика количества объектов по классам для слоя эволюционировавших звезд (EVOLVED) с помощью функции plot_class_counts, которая принимает DataFrame
    df_evolved,
    "EVOLVED: число объектов по классам",
    "evolved_class_counts.png",
)
plot_feature_histograms(df, "all")                  # Построение гистограмм распределения признаков teff_gspphot, logg_gspphot и radius_gspphot для всех MKGF (ALL MKGF) с помощью функции plot_feature_histograms, которая принимает DataFrame df и имя для заголовка графика. Этот график поможет визуально оценить распределение каждого признака в данных для всех MKGF и понять, есть ли какие-либо особенности или аномалии в распределении признаков.
plot_feature_histograms(df_dwarfs, "dwarfs")        # Построение гистограмм распределения признаков teff_gspphot, logg_gspphot и radius_gspphot для слоя карликов (DWARFS) с помощью функции plot_feature_histograms, которая принимает DataFrame df_dwarfs и имя для заголовка графика. Этот график поможет визуально оценить распределение каждого признака в данных для слоя карликов (DWARFS) с logg >= 4.0, который будет использоваться для построения многомерной гауссовой модели, и понять, есть ли какие-либо особенности или аномалии в распределении признаков для карликов.
plot_boxplots_dwarfs(df_dwarfs)                     # Построение коробчатых диаграмм (boxplots) для признаков teff_gspphot, logg_gspphot и radius_gspphot для слоя карликов (DWARFS) с помощью функции plot_boxplots_dwarfs, которая принимает DataFrame df_dwarfs. Этот график поможет визуально оценить распределение признаков для слоя карликов (DWARFS) с logg >= 4.0, который будет использоваться для построения многомерной гауссовой модели, и понять, есть ли какие-либо выбросы или аномалии в данных для карликов.
plot_scatter_layers(df_dwarfs, df_evolved)          # Построение диаграмм рассеяния (scatter plots) для признаков teff_gspphot, logg_gspphot и radius_gspphot для слоя карликов (DWARFS) и слоя эволюционировавших звезд (EVOLVED) с помощью функции plot_scatter_layers, которая принимает DataFrame df_dwarfs и df_evolved. Этот график поможет визуально оценить взаимосвязи между признаками для карликов и эволюционировавших звезд, а также понять, есть ли какие-либо особенности или аномалии в данных для этих слоев.
plot_logg_radius_with_threshold(df)                 # Построение диаграммы рассеяния (scatter plot) для признаков logg_gspphot и radius_gspphot для всех MKGF (ALL MKGF) с помощью функции plot_logg_radius_with_threshold, которая принимает DataFrame df. Этот график поможет визуально оценить взаимосвязь между логарифмом гравитационного ускорения (logg_gspphot) и радиусом (radius_gspphot) для всех MKGF, а также понять, есть ли какие-либо особенности или аномалии в данных для всех MKGF, и как они соотносятся с пороговым значением logg = 4.0, которое разделяет карликов (DWARFS) и эволюционировавшие звезды (EVOLVED).
plot_correlation_heatmaps(df, df_dwarfs, df_evolved)  # Построение тепловых карт (heatmaps) для матрицы корреляций между признаками teff_gspphot, logg_gspphot и radius_gspphot для всех MKGF (ALL MKGF), слоя карликов (DWARFS) и слоя эволюционировавших звезд (EVOLVED) с помощью функции plot_correlation_heatmaps, которая принимает DataFrame df, df_dwarfs и df_evolved. Этот график поможет визуально оценить взаимосвязи между признаками для разных слоев данных, а также понять, есть ли какие-либо особенности или аномалии в корреляциях между признаками для всех MKGF, карликов и эволюционировавших звезд.
print("Графики сохранены в:", PLOTS_DIR)            #Вывод сообщения о том, что графики для всех MKGF, слоя карликов (DWARFS) и слоя эволюционировавших звезд (EVOLVED) сохранены в директории, указанной в переменной PLOTS_DIR, которая была определена ранее в коде. Этот вывод поможет понять, где находятся сохраненные графики для дальнейшего анализа и отчетности.


# ============================================================
# 10. МНОГОМЕРНАЯ ГАУССОВА МОДЕЛЬ (Dwarfs only)
# ============================================================

print(                                      # Вывод заголовка для раздела многомерной гауссовой модели, который будет построен только для слоя карликов (DWARFS) с logg >= 4.0, и который будет использоваться для оценки плотности распределения объектов в признаковом пространстве teff_gspphot, logg_gspphot и radius_gspphot. Этот раздел поможет сосредоточиться на анализе и построении модели для карликов, исключая объекты с logg < 4.0, которые относятся к эволюционировавшим звездам (EVOLVED), и понять характеристики распределения признаков для карликов в данных для всех MKGF.
    "\n=== DWARFS: mu и cov по классам "
    "(M/K/G/F), только logg>=4.0 ==="
)


# Считает mu и cov (3D)
# для одного класса/подкласса.
def calc_gauss_stats(df_part: pd.DataFrame, label: str) -> None:    # Определение функции calc_gauss_stats, которая принимает DataFrame df_part и строку label в качестве аргументов, и которая рассчитывает среднее значение (mu) и ковариационную матрицу (cov) для признаков teff_gspphot, logg_gspphot и radius_gspphot для одного класса или подкласса в данных для слоя карликов (DWARFS). Эта функция поможет понять характеристики распределения признаков для конкретного класса или подкласса в данных для всех MKGF.
    x: FloatArray = feature_frame(df_part).to_numpy(dtype=float)                     # Преобразование выбранных признаков teff_gspphot, logg_gspphot и radius_gspphot из DataFrame df_part в NumPy-массив x с помощью метода to_numpy() и указания типа данных float. Этот массив будет использоваться для расчета среднего значения (mu) и ковариационной матрицы (cov) для конкретного класса или подкласса в данных для слоя карликов (DWARFS).
    n = x.shape[0]                                                  # Количество объектов в данном классе или подклассе, которое определяется как количество строк в массиве x с помощью атрибута shape[0]. Этот показатель поможет понять, сколько объектов соответствует конкретному классу или подклассу в данных для слоя карликов (DWARFS), и оценить устойчивость расчетов среднего значения (mu) и ковариационной матрицы (cov) для этого класса или подкласса.

    if n < 5:                                        # Проверка на минимальное количество объектов для устойчивого расчета ковариационной матрицы (cov). Если количество объектов n меньше 5, то выводится предупреждение о том, что слишком мало объектов для устойчивого расчета cov, и функция возвращает None, не выполняя дальнейшие расчеты. Этот порог в 5 объектов является эмпирическим и может быть изменен в зависимости от требований к устойчивости модели и характеристик данных.
        print(                                       # Вывод предупреждения о том, что для данного класса или подкласса недостаточно объектов для устойчивого расчета ковариационной матрицы (cov), и указание количества объектов n в этом классе или подклассе. Этот вывод поможет понять, что результаты для этого класса или подкласса могут быть ненадежными из-за недостаточного количества данных, и что следует быть осторожным при интерпретации этих результатов.
            f"\n[{label}] Слишком мало объектов "   
            f"для устойчивой cov: n={n}"
        )
        return                                       # Выход из функции, если количество объектов n меньше 5, что означает, что результаты для этого класса или подкласса могут быть ненадежными из-за недостаточного количества данных, и что следует быть осторожным при интерпретации этих результатов.

    mu: FloatArray = x.mean(axis=0)                         # Расчет среднего значения (mu) для признаков teff_gspphot, logg_gspphot и radius_gspphot для данного класса или подкласса с помощью метода mean() для массива x, который вычисляет среднее значение по каждому столбцу (axis=0) и возвращает вектор mu, который представляет собой среднее значение для каждого признака в данном классе или подклассе.
    sigma: FloatArray = np.cov(x, rowvar=False, ddof=1)     # Расчет ковариационной матрицы (cov) для признаков teff_gspphot, logg_gspphot и radius_gspphot для данного класса или подкласса с помощью функции np.cov() из библиотеки NumPy, которая принимает массив x, параметр rowvar=False для указания, что переменные представлены в столбцах, и параметр ddof=1 для использования несмещенной оценки ковариационной матрицы. Результатом является матрица sigma, которая представляет собой ковариационную матрицу для данного класса или подкласса.

    det_sigma = float(np.linalg.det(sigma))     # Расчет определителя ковариационной матрицы (det(cov)) для данного класса или подкласса с помощью функции np.linalg.det() из библиотеки NumPy, которая принимает матрицу sigma и возвращает ее определитель. Этот показатель поможет понять, насколько ковариационная матрица является вырожденной (det(cov) близок к нулю) или устойчивой (det(cov) значительно больше нуля) для данного класса или подкласса.
    eigvals: FloatArray = np.linalg.eigvalsh(sigma)         # Расчет собственных значений ковариационной матрицы (eigenvalues(cov)) для данного класса или подкласса с помощью функции np.linalg.eigvalsh() из библиотеки NumPy, которая принимает симметричную матрицу sigma и возвращает ее собственные значения. Этот показатель поможет понять, насколько ковариационная матрица является положительно определенной (все eigenvalues > 0) для данного класса или подкласса, что является важным условием для многомерной гауссовой модели.
    cond = float(np.linalg.cond(sigma))         # Расчет числа обусловленности ковариационной матрицы (cond(cov)) для данного класса или подкласса с помощью функции np.linalg.cond() из библиотеки NumPy, которая принимает матрицу sigma и возвращает ее число обусловленности. Этот показатель поможет понять, насколько ковариационная матрица является устойчивой для данного класса или подкласса, где низкое значение cond указывает на устойчивость, а высокое значение cond может указывать на проблемы с численной стабильностью при использовании этой матрицы в модели.

    print(f"\n[{label}] n={n}")                 # Вывод заголовка для данного класса или подкласса с указанием количества объектов n, который поможет понять, сколько объектов соответствует этому классу или подклассу в данных для слоя карликов (DWARFS), и оценить устойчивость расчетов среднего значения (mu) и ковариационной матрицы (cov) для этого класса или подкласса.
    print("mu =", mu)                           # Вывод среднего значения (mu) для признаков teff_gspphot, logg_gspphot и radius_gspphot для данного класса или подкласса, который поможет понять центральную тенденцию распределения признаков для этого класса или подкласса в данных для слоя карликов (DWARFS).
    print("cov =\n", sigma)                     # Вывод ковариационной матрицы (cov) для признаков teff_gspphot, logg_gspphot и radius_gspphot для данного класса или подкласса, который поможет понять взаимосвязи между признаками и вариацию данных для этого класса или подкласса в данных для слоя карликов (DWARFS).
    print("det(cov) =", det_sigma)              # Вывод определителя ковариационной матрицы (det(cov)) для данного класса или подкласса, который поможет понять, насколько ковариационная матрица является вырожденной (det(cov) близок к нулю) или устойчивой (det(cov) значительно больше нуля) для этого класса или подкласса в данных для слоя карликов (DWARFS).
    print("eigenvalues(cov) =", eigvals)        # Вывод собственных значений ковариационной матрицы (eigenvalues(cov)) для данного класса или подкласса, который поможет понять, насколько ковариационная матрица является положительно определенной (все eigenvalues > 0) для этого класса или подкласса в данных для слоя карликов (DWARFS), что является важным условием для многомерной гауссовой модели.
    print("cond(cov) =", cond)                  # Вывод числа обусловленности ковариационной матрицы (cond(cov)) для данного класса или подкласса, который поможет понять, насколько ковариационная матрица является устойчивой для этого класса или подкласса в данных для слоя карликов (DWARFS), где низкое значение cond указывает на устойчивость, а высокое значение cond может указывать на проблемы с численной стабильностью при использовании этой матрицы в модели.
    print("PD (все eigenvalues > 0):", bool(np.all(eigvals > 0)))   # Вывод информации о том, является ли ковариационная матрица положительно определенной (PD) для данного класса или подкласса, что проверяется с помощью функции np.all() для массива eigenvals, которая возвращает True, если все eigenvalues > 0, и False в противном случае. Этот показатель поможет понять, подходит ли ковариационная матрица для использования в многомерной гауссовой модели для этого класса или подкласса в данных для слоя карликов (DWARFS).


for cls in ["M", "K", "G", "F"]:                # Цикл по спектральным классам M, K, G и F для слоя карликов (DWARFS) с logg >= 4.0, который будет использоваться для построения многомерной гауссовой модели. Этот цикл поможет рассчитать среднее значение (mu) и ковариационную матрицу (cov) для каждого из этих классов в данных для слоя карликов (DWARFS), и понять характеристики распределения признаков для каждого класса.
    part: pd.DataFrame = df_dwarfs[df_dwarfs["spec_class"] == cls]    # Выбор подмножества данных для данного спектрального класса cls (M, K, G или F) из DataFrame df_dwarfs, который содержит объекты с logg >= 4.0 (карлики), с помощью условия фильтрации df_dwarfs["spec_class"] == cls. Этот выбор поможет сосредоточиться на анализе и расчете статистических показателей для конкретного класса в данных для слоя карликов (DWARFS).
    calc_gauss_stats(part, f"CLASS {cls}")              # Вызов функции calc_gauss_stats для данного спектрального класса cls (M, K, G или F) с передачей подмножества данных part и строки label, которая указывает на класс, для расчета среднего значения (mu) и ковариационной матрицы (cov) для этого класса в данных для слоя карликов (DWARFS), и понимания характеристик распределения признаков для этого класса.


# ============================================================
# 11. ПОДКЛАССЫ M-КАРЛИКОВ (Early / Mid / Late)
# ============================================================

print(                                          # Вывод заголовка для раздела подклассов M-карликов (Early/Mid/Late), который поможет сосредоточиться на анализе и расчете статистических показателей для подклассов M-карликов в данных для слоя карликов (DWARFS) с logg >= 4.0, и понять характеристики распределения признаков для этих подклассов в данных для всех MKGF.
    "\n=== DWARFS: mu и cov "
    "для подклассов M (Early/Mid/Late) ==="
)

df_m: pd.DataFrame = df_dwarfs[df_dwarfs["spec_class"] == "M"].copy()     # Выбор подмножества данных для спектрального класса M из DataFrame df_dwarfs, который содержит объекты с logg >= 4.0 (карлики), с помощью условия фильтрации df_dwarfs["spec_class"] == "M", и сохранение этого подмножества в переменную df_m для дальнейшего анализа и расчета статистических показателей для подклассов M-карликов (Early/Mid/Late) в данных для слоя карликов (DWARFS).

m_early: pd.DataFrame = df_m[                                    # Выбор подмножества данных для подкласса M_EARLY из DataFrame df_m, который содержит объекты спектрального класса M, с помощью условия фильтрации по признаку teff_gspphot, который определяет границы для подкласса M_EARLY (от 3500K до 4000K). Этот выбор поможет сосредоточиться на анализе и расчете статистических показателей для подкласса M_EARLY в данных для слоя карликов (DWARFS) с logg >= 4.0, и понять характеристики распределения признаков для этого подкласса в данных для всех MKGF.
    (df_m["teff_gspphot"] >= M_EARLY_MIN)          # Условие для выбора объектов, которые принадлежат к подклассу M_EARLY, где teff_gspphot должен быть больше или равен M_EARLY_MIN (3500K) и меньше M_EARLY_MAX (4000K). Этот выбор поможет сосредоточиться на анализе и расчете статистических показателей для подкласса M_EARLY в данных для слоя карликов (DWARFS) с logg >= 4.0, и понять характеристики распределения признаков для этого подкласса в данных для всех MKGF.
    & (df_m["teff_gspphot"] < M_EARLY_MAX)         # Условие для выбора объектов, которые принадлежат к подклассу M_EARLY, где teff_gspphot должен быть меньше M_EARLY_MAX (4000K) и больше или равен M_EARLY_MIN (3500K). Этот выбор поможет сосредоточиться на анализе и расчете статистических показателей для подкласса M_EARLY в данных для слоя карликов (DWARFS) с logg >= 4.0, и понять характеристики распределения признаков для этого подкласса в данных для всех MKGF.
]
m_mid: pd.DataFrame = df_m[                                       # Выбор подмножества данных для подкласса M_MID из DataFrame df_m, который содержит объекты спектрального класса M, с помощью условия фильтрации по признаку teff_gspphot, который определяет границы для подкласса M_MID (от 3200K до 3500K). Этот выбор поможет сосредоточиться на анализе и расчете статистических показателей для подкласса M_MID в данных для слоя карликов (DWARFS) с logg >= 4.0, и понять характеристики распределения признаков для этого подкласса в данных для всех MKGF.
    (df_m["teff_gspphot"] >= M_MID_MIN)             # Условие для выбора объектов, которые принадлежат к подклассу M_MID, где teff_gspphot должен быть больше или равен M_MID_MIN (3200K) и меньше M_MID_MAX (3500K). Этот выбор поможет сосредоточиться на анализе и расчете статистических показателей для подкласса M_MID в данных для слоя карликов (DWARFS) с logg >= 4.0, и понять характеристики распределения признаков для этого подкласса в данных для всех MKGF.
    & (df_m["teff_gspphot"] < M_EARLY_MIN)          # Условие для выбора объектов, которые принадлежат к подклассу M_MID, где teff_gspphot должен быть меньше M_EARLY_MIN (3500K) и больше или равен M_MID_MIN (3200K). Этот выбор поможет сосредоточиться на анализе и расчете статистических показателей для подкласса M_MID в данных для слоя карликов (DWARFS) с logg >= 4.0, и понять характеристики распределения признаков для этого подкласса в данных для всех MKGF.
]
m_late: pd.DataFrame = df_m[df_m["teff_gspphot"] < M_MID_MIN]     # Выбор подмножества данных для подкласса M_LATE из DataFrame df_m, который содержит объекты спектрального класса M, с помощью условия фильтрации по признаку teff_gspphot, который определяет границу для подкласса M_LATE (меньше 3200K). Этот выбор поможет сосредоточиться на анализе и расчете статистических показателей для подкласса M_LATE в данных для слоя карликов (DWARFS) с logg >= 4.0, и понять характеристики распределения признаков для этого подкласса в данных для всех MKGF.

calc_gauss_stats(m_early, "M_EARLY [3500, 4000)")   # Вызов функции calc_gauss_stats для подкласса M_EARLY с передачей подмножества данных m_early и строки label, которая указывает на подкласс, для расчета среднего значения (mu) и ковариационной матрицы (cov) для этого подкласса в данных для слоя карликов (DWARFS), и понимания характеристик распределения признаков для этого подкласса в данных для всех MKGF.
calc_gauss_stats(m_mid, "M_MID (3200-3500K)")       # Вызов функции calc_gauss_stats для подкласса M_MID с передачей подмножества данных m_mid и строки label, которая указывает на подкласс, для расчета среднего значения (mu) и ковариационной матрицы (cov) для этого подкласса в данных для слоя карликов (DWARFS), и понимания характеристик распределения признаков для этого подкласса в данных для всех MKGF.
calc_gauss_stats(m_late, "M_LATE (<3200K)")         # Вызов функции calc_gauss_stats для подкласса M_LATE с передачей подмножества данных m_late и строки label, которая указывает на подкласс, для расчета среднего значения (mu) и ковариационной матрицы (cov) для этого подкласса в данных для слоя карликов (DWARFS), и понимания характеристик распределения признаков для этого подкласса в данных для всех MKGF.

