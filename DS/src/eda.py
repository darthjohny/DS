# eda.py
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false
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

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine


# ============================================================
# 1. ПАРАМЕТРЫ ПОДКЛЮЧЕНИЯ К БД
# ============================================================

USER = "postgres"
PASSWORD = "1234"  # пароль как в DBeaver
HOST = "127.0.0.1"
PORT = 5432
DB = "dspro_vkr_research"

engine = create_engine(
    f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"
)

sns.set_theme(style="whitegrid")

CLASS_ORDER = ["M", "K", "G", "F"]
FEATURES = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
PLOTS_DIR = "data/eda/plots"
LOGG_DWARF_MIN = 4.0

# Пороги подклассов M.
# Верхняя граница M_EARLY
# делается строгой (< 4000),
# чтобы не пересекаться с K-областью.
M_EARLY_MIN = 3500.0
M_EARLY_MAX = 4000.0
M_MID_MIN = 3200.0


def save_plot(filename: str) -> None:
    """Сохраняет текущий matplotlib-график."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_class_counts(
    df_part: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    """График количества объектов
    по классам.
    """
    _, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(
        data=df_part,
        x="spec_class",
        # hue = x нужен для актуального API seaborn,
        # иначе будут предупреждения
        # о deprecated-поведении.
        hue="spec_class",
        order=CLASS_ORDER,
        hue_order=CLASS_ORDER,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("spec_class")
    ax.set_ylabel("count")
    save_plot(filename)


def plot_feature_histograms(
    df_part: pd.DataFrame,
    prefix: str,
) -> None:
    """Гистограммы признаков по классам."""
    for feature in FEATURES:
        _, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=df_part,
            x=feature,
            hue="spec_class",
            hue_order=CLASS_ORDER,
            bins=30,
            stat="density",
            common_norm=False,
            alpha=0.35,
            kde=True,
            ax=ax,
        )
        ax.set_title(f"{prefix}: распределение {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        save_plot(f"{prefix}_{feature}_hist.png")


def plot_boxplots_dwarfs(df_part: pd.DataFrame) -> None:
    """Boxplot признаков для слоя карликов."""
    for feature in FEATURES:
        _, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df_part,
            x="spec_class",
            y=feature,
            # Аналогично countplot:
            # hue = x для стабильного поведения seaborn.
            hue="spec_class",
            order=CLASS_ORDER,
            hue_order=CLASS_ORDER,
            palette="Set2",
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"DWARFS: boxplot {feature}")
        ax.set_xlabel("spec_class")
        ax.set_ylabel(feature)
        save_plot(f"dwarfs_{feature}_boxplot.png")


def plot_scatter_layers(
    df_dwarfs_part: pd.DataFrame,
    df_evolved_part: pd.DataFrame,
) -> None:
    """Scatter для сравнения слоёв DWARFS и EVOLVED."""
    _, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df_dwarfs_part["teff_gspphot"],
        df_dwarfs_part["radius_gspphot"],
        s=12,
        alpha=0.45,
        label="DWARFS",
    )
    ax.scatter(
        df_evolved_part["teff_gspphot"],
        df_evolved_part["radius_gspphot"],
        s=12,
        alpha=0.45,
        label="EVOLVED",
    )
    ax.set_title("Teff vs Radius: DWARFS и EVOLVED")
    ax.set_xlabel("teff_gspphot")
    ax.set_ylabel("radius_gspphot")
    ax.legend()
    save_plot("layers_teff_vs_radius_scatter.png")


def plot_logg_radius_with_threshold(df_part: pd.DataFrame) -> None:
    """Scatter logg-radius и порог карликов logg = 4.0."""
    _, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_part,
        x="logg_gspphot",
        y="radius_gspphot",
        hue="spec_class",
        hue_order=CLASS_ORDER,
        alpha=0.55,
        s=18,
        ax=ax,
    )
    ax.axvline(
        x=LOGG_DWARF_MIN,
        color="red",
        linestyle="--",
        linewidth=1.4,
        label=f"logg = {LOGG_DWARF_MIN}",
    )
    ax.set_title("ALL MKGF: logg vs radius")
    ax.set_xlabel("logg_gspphot")
    ax.set_ylabel("radius_gspphot")
    ax.legend()
    save_plot("all_logg_vs_radius_with_threshold.png")


def plot_correlation_heatmaps(
    df_all: pd.DataFrame,
    df_dwarfs_part: pd.DataFrame,
    df_evolved_part: pd.DataFrame,
) -> None:
    """Heatmap корреляций для ALL / DWARFS / EVOLVED."""
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    layers = [
        ("ALL", df_all),
        ("DWARFS", df_dwarfs_part),
        ("EVOLVED", df_evolved_part),
    ]
    for idx, (name, layer_df) in enumerate(layers):
        corr = layer_df[FEATURES].corr()
        sns.heatmap(
            corr,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            square=True,
            cbar=idx == 2,
            ax=axes[idx],
        )
        axes[idx].set_title(name)
    save_plot("corr_heatmaps_all_dwarfs_evolved.png")


# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ ДЛЯ EDA
# ============================================================

# Источники:
# - lab.v_nasa_gaia_train_classified : MKGF хосты с spec_class
# - lab.v_nasa_gaia_train_dwarfs     : MKGF, только logg >= 4
# - lab.v_nasa_gaia_train_evolved    : MKGF, только logg < 4

# Базовая обучающая выборка
# (все MKGF, включая evolved)
QUERY_ALL_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M','K','G','F');
"""

# Карлики для обучения гауссовской модели
QUERY_DWARFS_MKGF = """
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
QUERY_EVOLVED_MKGF = """
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

df = pd.read_sql(QUERY_ALL_MKGF, engine)
df_dwarfs = pd.read_sql(QUERY_DWARFS_MKGF, engine)
df_evolved = pd.read_sql(QUERY_EVOLVED_MKGF, engine)

print("\n=== ДАННЫЕ ЗАГРУЖЕНЫ ===")
print("Размер ALL MKGF:", df.shape)
print("Размер DWARFS (logg>=4.0):", df_dwarfs.shape)
print("Размер EVOLVED (logg<4.0):", df_evolved.shape)


# ============================================================
# 3. ПРОВЕРКА NULL (ALL MKGF)
# ============================================================

print("\n=== ПРОВЕРКА NULL ===")
print("NULL по столбцам (ALL MKGF):\n", df.isnull().sum())


# ============================================================
# 4. ОБЩИЕ СТАТИСТИКИ (ALL MKGF)
# ============================================================

print("\n=== ОБЩАЯ СТАТИСТИКА (describe) ===")
print(df.describe())


# ============================================================
# 5. КОРРЕЛЯЦИИ (ALL MKGF)
# ============================================================

print("\n=== КОРРЕЛЯЦИИ МЕЖДУ ПРИЗНАКАМИ ===")
print(df[["teff_gspphot", "logg_gspphot", "radius_gspphot"]].corr())


# ============================================================
# 6. СТАТИСТИКА ПО КЛАССАМ (ALL MKGF: M/K/G/F)
# ============================================================

print(
    "\n=== СТАТИСТИКА "
    "ПО СПЕКТРАЛЬНЫМ КЛАССАМ ==="
)

group_stats = (
    df.groupby("spec_class")[
        ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
    ].agg(["mean", "std", "min", "max"])
)

print(group_stats)


# ============================================================
# 7. БЫСТРАЯ ПРОВЕРКА EVOLVED НА ALL MKGF
# ============================================================

print(
    "\n=== ALL MKGF: ТОП-20 ПО РАДИУСУ "
    "(быстрая проверка EVOLVED/гигантов) ==="
)

top_radius = df.sort_values("radius_gspphot", ascending=False).head(20)
print(top_radius)


# ============================================================
# 8. СЛОЙ DWARFS (ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ)
# ============================================================

print(
    "\n=== ТОЛЬКО ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ "
    "(из view: v_nasa_gaia_train_dwarfs) ==="
)
print("Размер выборки:", df_dwarfs.shape)

print("\n=== СТАТИСТИКА ПОСЛЕ ФИЛЬТРА ===")
print(df_dwarfs.describe())

print("\n=== КОРРЕЛЯЦИИ ПОСЛЕ ФИЛЬТРА ===")
print(df_dwarfs[["teff_gspphot", "logg_gspphot", "radius_gspphot"]].corr())


# ============================================================
# 9. СЛОЙ EVOLVED (logg < 4.0)
# ============================================================

# Эти объекты не входят
# в гауссову модель карликов,
# но сохраняются для аналитики и отчёта.
df_anomalies = df_evolved.copy()

print(
    "\n=== EVOLVED (logg < 4.0): "
    "отдельный аналитический слой ==="
)
print("Количество:", df_anomalies.shape)

os.makedirs("data/eda", exist_ok=True)
df_anomalies.to_csv("data/eda/evolved_stars_snapshot.csv", index=False)


# ============================================================
# 9.1. СЛОЙ A/B/O REF (не хосты)
# ============================================================

print(
    "\n=== A/B/O REF (не хосты): "
    "диапазоны v_gaia_ref_abo_training ==="
)

QUERY_ABO_REF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_gaia_ref_abo_training
WHERE spec_class IN ('A','B','O');
"""

df_abo = pd.read_sql(QUERY_ABO_REF, engine)
print("Размер ABO ref:", df_abo.shape)
print("\n=== ABO ref: describe ===")
print(df_abo.describe())

print("\n=== ABO ref: статистика по классам ===")
abo_stats = (
    df_abo.groupby("spec_class")[FEATURES]
    .agg(["count", "mean", "std", "min", "max"])
)
print(abo_stats)

df_abo.sort_values("teff_gspphot", ascending=False).head(20).to_csv(
    "data/eda/abo_top20_by_teff.csv",
    index=False,
)


# ============================================================
# 9.2. ВИЗУАЛИЗАЦИЯ EDA
# ============================================================

print(
    "\n=== ВИЗУАЛИЗАЦИЯ EDA: "
    "сохраняем графики ==="
)

plot_class_counts(
    df,
    "ALL MKGF: число объектов по классам",
    "all_class_counts.png",
)
plot_class_counts(
    df_dwarfs,
    "DWARFS: число объектов по классам",
    "dwarfs_class_counts.png",
)
plot_class_counts(
    df_evolved,
    "EVOLVED: число объектов по классам",
    "evolved_class_counts.png",
)
plot_feature_histograms(df, "all")
plot_feature_histograms(df_dwarfs, "dwarfs")
plot_boxplots_dwarfs(df_dwarfs)
plot_scatter_layers(df_dwarfs, df_evolved)
plot_logg_radius_with_threshold(df)
plot_correlation_heatmaps(df, df_dwarfs, df_evolved)

print("Графики сохранены в:", PLOTS_DIR)


# ============================================================
# 10. МНОГОМЕРНАЯ ГАУССОВА МОДЕЛЬ (Dwarfs only)
# ============================================================

print(
    "\n=== DWARFS: mu и cov по классам "
    "(M/K/G/F), только logg>=4.0 ==="
)


# Считает mu и cov (3D)
# для одного класса/подкласса.
def calc_gauss_stats(df_part: pd.DataFrame, label: str) -> None:
    x = df_part[FEATURES].to_numpy(dtype=float)
    n = x.shape[0]

    if n < 5:
        print(
            f"\n[{label}] Слишком мало объектов "
            f"для устойчивой cov: n={n}"
        )
        return

    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False, ddof=1)

    det_sigma = float(np.linalg.det(sigma))
    eigvals = np.linalg.eigvalsh(sigma)
    cond = float(np.linalg.cond(sigma))

    print(f"\n[{label}] n={n}")
    print("mu =", mu)
    print("cov =\n", sigma)
    print("det(cov) =", det_sigma)
    print("eigenvalues(cov) =", eigvals)
    print("cond(cov) =", cond)
    print("PD (все eigenvalues > 0):", bool(np.all(eigvals > 0)))


for cls in ["M", "K", "G", "F"]:
    part = df_dwarfs[df_dwarfs["spec_class"] == cls]
    calc_gauss_stats(part, f"CLASS {cls}")


# ============================================================
# 11. ПОДКЛАССЫ M-КАРЛИКОВ (Early / Mid / Late)
# ============================================================

print(
    "\n=== DWARFS: mu и cov "
    "для подклассов M (Early/Mid/Late) ==="
)

df_m = df_dwarfs[df_dwarfs["spec_class"] == "M"].copy()

m_early = df_m[
    (df_m["teff_gspphot"] >= M_EARLY_MIN)
    & (df_m["teff_gspphot"] < M_EARLY_MAX)
]
m_mid = df_m[
    (df_m["teff_gspphot"] >= M_MID_MIN)
    & (df_m["teff_gspphot"] < M_EARLY_MIN)
]
m_late = df_m[df_m["teff_gspphot"] < M_MID_MIN]

calc_gauss_stats(m_early, "M_EARLY [3500, 4000)")
calc_gauss_stats(m_mid, "M_MID (3200-3500K)")
calc_gauss_stats(m_late, "M_LATE (<3200K)")


# ============================================================
# 12. МИКРО-ИТОГ EDA
# ============================================================
# - EVOLVED (logg < 4.0)
#   искажают радиусы и cov для карликов.
# - Это не ошибка данных,
#   а отдельная физическая популяция.
# - Для модели карликов
#   используем только слой DWARFS.
# - Для классов M/K/G/F
#   рассчитаны mu, cov, det, eigenvalues, cond.
# - Следующий шаг:
#   нормализация + shrinkage
#   перед финальным скорингом.
# - A/B/O храним отдельно
#   как reference-слой для sanity-check и OOD.
# ============================================================
