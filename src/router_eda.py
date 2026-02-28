# router_eda.py
# ============================================================
# Назначение
# ------------------------------------------------------------
# EDA для Gaussian-router в проекте ВКР
# (Gaia DR3 + NASA hosts).
#
# Цель файла:
# проверить и визуально зафиксировать,
# что reference-слой для router-модели
# физически согласован
# и пригоден для Gaussian / Mahalanobis подхода.
#
# Что делает этот файл:
# 1. Загружает единый router-reference слой
#    из lab.v_gaia_router_training.
# 2. Проверяет полноту данных
#    по ключевым признакам
#    teff_gspphot / logg_gspphot / radius_gspphot.
# 3. Формирует router_label
#    вида A_dwarf, K_evolved и т.д.
# 4. Считает базовые статистики
#    и размеры классов.
# 5. Строит графики распределений,
#    scatter и heatmap.
# 6. Сохраняет snapshot и таблицу Gaussian readiness.
#
# Важно:
# - здесь нет обучения router-модели;
# - здесь нет host-like similarity;
# - здесь нет ranking layer;
# - здесь нет записи результатов
#   распознавания в БД.
#
# То есть:
# router_eda.py отвечает только
# за исследование и валидацию
# reference-слоя для gaussian_router.py.
# ============================================================

"""EDA и валидация router-reference слоя."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

FEATURES: List[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
SPEC_CLASSES: List[str] = ["A", "B", "F", "G", "K", "M", "O"]
EVOLUTION_STAGES: List[str] = ["dwarf", "evolved"]
ROUTER_VIEW = "lab.v_gaia_router_training"
OUTPUT_DIR = Path("data/router_eda")
PLOTS_DIR = OUTPUT_DIR / "plots"
SNAPSHOT_PATH = OUTPUT_DIR / "router_training_snapshot.csv"
CLASS_COUNTS_PATH = OUTPUT_DIR / "router_class_counts.csv"
GAUSS_STATS_PATH = OUTPUT_DIR / "router_gaussian_readiness.csv"

FloatArray = npt.NDArray[np.floating[Any]]


def _load_dotenv_local(path: str = ".env") -> None:
    """Load .env values into the process environment without overwriting."""
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as file:
            for raw in file:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def make_engine_from_env() -> Engine:
    """Create SQLAlchemy engine from DATABASE_URL or PG* variables."""
    _load_dotenv_local(".env")
    url = os.getenv("DATABASE_URL")

    if url:
        bad_tokens = ["HOST", "USER", "PASSWORD", "DBNAME"]
        if any(token in url for token in bad_tokens):
            raise RuntimeError(
                "DATABASE_URL looks like a placeholder. Provide a real DSN."
            )
        return create_engine(url)

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if all([host, dbname, user, password]):
        return create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        )

    raise RuntimeError(
        "Database connection is missing. Set DATABASE_URL or PG* variables."
    )


def read_sql_frame(engine: Engine, query: str) -> pd.DataFrame:
    """Typed wrapper around pandas.read_sql for local EDA queries."""
    return pd.read_sql(query, engine)


def feature_frame(df_part: pd.DataFrame) -> pd.DataFrame:
    """Return the core feature subset with explicit DataFrame typing."""
    return df_part[FEATURES]


def make_router_label(spec_class: Any, evolution_stage: Any) -> str:
    """Build a stable router label from class and stage."""
    spec_part = str(spec_class).strip().upper()
    stage_part = str(evolution_stage).strip().lower()
    return f"{spec_part}_{stage_part}"


def ensure_router_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add router_label column if it does not exist yet."""
    result = df.copy()
    result["router_label"] = [
        make_router_label(
            spec_class=spec_class,
            evolution_stage=evolution_stage,
        )
        for spec_class, evolution_stage in zip(
            result["spec_class"], result["evolution_stage"]
        )
    ]
    return result


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


def save_plot(filename: str, figure: Figure) -> None:
    """Save a matplotlib figure into the router EDA plot directory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    figure_any: Any = figure
    plt_any: Any = plt
    figure_any.tight_layout()
    figure_any.savefig(path, dpi=160)
    plt_any.close(figure)


def print_basic_summary(df_router: pd.DataFrame) -> None:
    """Print basic sanity-check information for the router dataset."""
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
    """Print NULL counts for the router dataset."""
    columns = ["source_id", "spec_class", "evolution_stage"] + FEATURES
    print("\n=== NULL REPORT ===")
    print(df_router[columns].isnull().sum())


def print_router_describe(df_router: pd.DataFrame) -> None:
    """Print descriptive statistics for the core router features."""
    print("\n=== DESCRIBE (ROUTER FEATURES) ===")
    print(feature_frame(df_router).describe())


def build_class_counts(df_router: pd.DataFrame) -> pd.DataFrame:
    """Aggregate class counts for spec_class and evolution_stage."""
    counts = (
        df_router.groupby(
            ["spec_class", "evolution_stage", "router_label"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_objects"})
        .sort_values(
            ["spec_class", "evolution_stage", "router_label"],
            ignore_index=True,
        )
    )
    return counts


def print_class_counts(counts: pd.DataFrame) -> None:
    """Print class counts for the router dataset."""
    print("\n=== CLASS COUNTS ===")
    print(counts.to_string(index=False))


def is_missing_scalar(value: Any) -> bool:
    """Return True for None / NaN / pd.NA."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def has_missing_values(values: Iterable[Any]) -> bool:
    """Return True if any feature is missing."""
    return any(is_missing_scalar(value) for value in values)


def calc_router_gauss_stats(df_router: pd.DataFrame) -> pd.DataFrame:
    """Calculate Gaussian readiness statistics for each router_label."""
    rows: list[dict[str, Any]] = []

    for router_label, subset in df_router.groupby("router_label", sort=True):
        x: FloatArray = feature_frame(subset).to_numpy(dtype=float)
        n = int(x.shape[0])

        if n < 5:
            rows.append(
                {
                    "router_label": str(router_label),
                    "spec_class": str(subset["spec_class"].iloc[0]),
                    "evolution_stage": str(subset["evolution_stage"].iloc[0]),
                    "n_objects": n,
                    "mu_teff": float("nan"),
                    "mu_logg": float("nan"),
                    "mu_radius": float("nan"),
                    "det_cov": float("nan"),
                    "cond_cov": float("nan"),
                    "pd_cov": False,
                    "corr_teff_logg": float("nan"),
                    "corr_teff_radius": float("nan"),
                    "corr_logg_radius": float("nan"),
                }
            )
            continue

        mu: FloatArray = x.mean(axis=0)
        sigma: FloatArray = np.cov(x, rowvar=False, ddof=1)
        det_sigma = float(np.linalg.det(sigma))
        eigvals: FloatArray = np.linalg.eigvalsh(sigma)
        cond = float(np.linalg.cond(sigma))
        corr = feature_frame(subset).corr()
        mu_any: Any = mu
        corr_any: Any = corr

        rows.append(
            {
                "router_label": str(router_label),
                "spec_class": str(subset["spec_class"].iloc[0]),
                "evolution_stage": str(subset["evolution_stage"].iloc[0]),
                "n_objects": n,
                "mu_teff": float(mu_any[0]),
                "mu_logg": float(mu_any[1]),
                "mu_radius": float(mu_any[2]),
                "det_cov": det_sigma,
                "cond_cov": cond,
                "pd_cov": bool(np.all(eigvals > 0.0)),
                "corr_teff_logg": float(
                    corr_any.loc[FEATURES[0], FEATURES[1]]
                ),
                "corr_teff_radius": float(
                    corr_any.loc[FEATURES[0], FEATURES[2]]
                ),
                "corr_logg_radius": float(
                    corr_any.loc[FEATURES[1], FEATURES[2]]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("router_label", ignore_index=True)


def print_gaussian_readiness(stats: pd.DataFrame) -> None:
    """Print a compact Gaussian readiness summary."""
    print("\n=== GAUSSIAN READINESS ===")
    preview_cols = [
        "router_label",
        "n_objects",
        "det_cov",
        "cond_cov",
        "pd_cov",
    ]
    print(stats[preview_cols].to_string(index=False))


def plot_router_label_counts(df_router: pd.DataFrame) -> None:
    """Plot class counts by spec_class and evolution_stage."""
    figure, ax = make_figure_ax((9, 5))
    draw_countplot(
        data=df_router,
        x="spec_class",
        hue="evolution_stage",
        order=SPEC_CLASSES,
        hue_order=EVOLUTION_STAGES,
        palette="Set2",
        ax=ax,
    )
    set_axes_title(
        ax,
        "Router reference: число объектов "
        "по классам и стадиям",
    )
    set_axes_xlabel(ax, "spec_class")
    set_axes_ylabel(ax, "count")
    save_plot("router_class_stage_counts.png", figure)


def plot_router_feature_histograms(df_router: pd.DataFrame) -> None:
    """Plot feature histograms split by evolution stage."""
    for feature in FEATURES:
        figure, ax = make_figure_ax((8, 5))
        draw_histplot(
            data=df_router,
            x=feature,
            hue="evolution_stage",
            hue_order=EVOLUTION_STAGES,
            bins=35,
            stat="density",
            common_norm=False,
            alpha=0.35,
            kde=True,
            ax=ax,
        )
        set_axes_title(
            ax,
            f"Router reference: распределение {feature}",
        )
        set_axes_xlabel(ax, feature)
        set_axes_ylabel(ax, "density")
        save_plot(f"router_{feature}_hist.png", figure)


def plot_router_feature_boxplots(df_router: pd.DataFrame) -> None:
    """Plot boxplots by class and evolution stage."""
    for feature in FEATURES:
        figure, ax = make_figure_ax((10, 5))
        draw_boxplot(
            data=df_router,
            x="spec_class",
            y=feature,
            hue="evolution_stage",
            order=SPEC_CLASSES,
            hue_order=EVOLUTION_STAGES,
            palette="Set3",
            ax=ax,
        )
        set_axes_title(ax, f"Router reference: boxplot {feature}")
        set_axes_xlabel(ax, "spec_class")
        set_axes_ylabel(ax, feature)
        save_plot(f"router_{feature}_boxplot.png", figure)


def plot_router_scatter(
    df_router: pd.DataFrame,
    *,
    x: str,
    y: str,
    filename: str,
    title: str,
) -> None:
    """Plot one scatter chart for the router dataset."""
    figure, ax = make_figure_ax((9, 6))
    draw_scatterplot(
        data=df_router,
        x=x,
        y=y,
        hue="spec_class",
        style="evolution_stage",
        hue_order=SPEC_CLASSES,
        style_order=EVOLUTION_STAGES,
        alpha=0.55,
        s=26,
        ax=ax,
    )
    set_axes_title(ax, title)
    set_axes_xlabel(ax, x)
    set_axes_ylabel(ax, y)
    save_plot(filename, figure)


def plot_router_correlation_heatmaps(df_router: pd.DataFrame) -> None:
    """Plot heatmaps for ALL / DWARFS / EVOLVED router layers."""
    figure, axes = make_figure_axes(3, (15, 4))
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
        set_axes_title(ax, name)

    save_plot("router_corr_heatmaps_all_dwarfs_evolved.png", figure)


def save_router_artifacts(
    df_router: pd.DataFrame,
    counts: pd.DataFrame,
    gauss_stats: pd.DataFrame,
) -> None:
    """Save router EDA tables for later inspection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_router.to_csv(SNAPSHOT_PATH, index=False)
    counts.to_csv(CLASS_COUNTS_PATH, index=False)
    gauss_stats.to_csv(GAUSS_STATS_PATH, index=False)


def main() -> None:
    """Run router EDA end-to-end."""
    engine = make_engine_from_env()
    query = f"""
    SELECT
        source_id,
        spec_class,
        evolution_stage,
        {", ".join(FEATURES)}
    FROM {ROUTER_VIEW}
    WHERE spec_class IN ('A','B','F','G','K','M','O')
      AND evolution_stage IN ('dwarf','evolved');
    """

    df_router = ensure_router_labels(read_sql_frame(engine, query))
    counts = build_class_counts(df_router)
    gauss_stats = calc_router_gauss_stats(df_router)

    print_basic_summary(df_router)
    print_null_report(df_router)
    print_router_describe(df_router)
    print_class_counts(counts)
    print_gaussian_readiness(gauss_stats)

    save_router_artifacts(df_router, counts, gauss_stats)
    plot_router_label_counts(df_router)
    plot_router_feature_histograms(df_router)
    plot_router_feature_boxplots(df_router)
    plot_router_scatter(
        df_router,
        x="teff_gspphot",
        y="logg_gspphot",
        filename="router_teff_vs_logg_scatter.png",
        title="Router reference: Teff vs logg",
    )
    plot_router_scatter(
        df_router,
        x="teff_gspphot",
        y="radius_gspphot",
        filename="router_teff_vs_radius_scatter.png",
        title="Router reference: Teff vs radius",
    )
    plot_router_correlation_heatmaps(df_router)

    print("\n=== FILES SAVED ===")
    print("Snapshot:", SNAPSHOT_PATH)
    print("Class counts:", CLASS_COUNTS_PATH)
    print("Gaussian readiness:", GAUSS_STATS_PATH)
    print("Plots dir:", PLOTS_DIR)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
