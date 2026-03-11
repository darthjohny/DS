"""CLI-точка входа для host EDA."""

from __future__ import annotations

import os

import pandas as pd

from analysis.host_eda.constants import (
    ABO_TOP20_PATH,
    CONTRASTIVE_GAUSS_STATS_PATH,
    CONTRASTIVE_MIN_POPULATION_SIZE,
    CONTRASTIVE_SHRINK_ALPHA,
    CONTRASTIVE_VIEW_ENV,
    EVOLVED_SNAPSHOT_PATH,
    M_EARLY_MAX,
    M_EARLY_MIN,
    M_MID_MIN,
)
from analysis.host_eda.contrastive import (
    calc_contrastive_gauss_stats,
    print_contrastive_gauss_stats,
)
from analysis.host_eda.data import (
    get_engine,
    load_abo_ref,
    load_all_mkgf,
    load_dwarfs_mkgf,
    load_evolved_mkgf,
)
from analysis.host_eda.exports import save_frame
from analysis.host_eda.plots import (
    plot_boxplots_dwarfs,
    plot_class_counts,
    plot_correlation_heatmaps,
    plot_feature_histograms,
    plot_logg_radius_with_threshold,
    plot_scatter_layers,
    set_default_theme,
)
from analysis.host_eda.stats import (
    build_group_stats,
    calc_gauss_stats,
    top_by_radius,
    top_by_teff,
)
from model_gaussian import (
    CONTRASTIVE_POPULATION_COLUMN,
    load_contrastive_training_from_db,
)


def main() -> None:
    """Запустить полный host-layer EDA сценарий end-to-end.

    Сценарий загружает выборки ALL/DWARFS/EVOLVED и A/B/O reference,
    печатает статистические сводки, сохраняет ключевые DataFrame и
    строит графики в `data/eda/host`. При наличии
    `CONTRASTIVE_HOST_FIELD_VIEW` дополнительно выполняется contrastive
    readiness-анализ.
    """
    set_default_theme()
    df = load_all_mkgf()
    df_dwarfs = load_dwarfs_mkgf()
    df_evolved = load_evolved_mkgf()

    print("\n=== ДАННЫЕ ЗАГРУЖЕНЫ ===")
    print("Размер ALL MKGF:", df.shape)
    print("Размер DWARFS (logg>=4.0):", df_dwarfs.shape)
    print("Размер EVOLVED (logg<4.0):", df_evolved.shape)

    print("\n=== ПРОВЕРКА NULL ===")
    print("NULL по столбцам (ALL MKGF):\n", df.isnull().sum())

    print("\n=== ОБЩАЯ СТАТИСТИКА (describe) ===")
    print(df.describe())

    print("\n=== КОРРЕЛЯЦИИ МЕЖДУ ПРИЗНАКАМИ ===")
    print(df[["teff_gspphot", "logg_gspphot", "radius_gspphot"]].corr())

    print("\n=== СТАТИСТИКА ПО СПЕКТРАЛЬНЫМ КЛАССАМ ===")
    print(build_group_stats(df))

    print("\n=== ALL MKGF: ТОП-20 ПО РАДИУСУ (быстрая проверка EVOLVED/гигантов) ===")
    print(top_by_radius(df))

    print("\n=== ТОЛЬКО ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ (из view: v_nasa_gaia_train_dwarfs) ===")
    print("Размер выборки:", df_dwarfs.shape)

    print("\n=== СТАТИСТИКА ПОСЛЕ ФИЛЬТРА ===")
    print(df_dwarfs.describe())

    print("\n=== КОРРЕЛЯЦИИ ПОСЛЕ ФИЛЬТРА ===")
    print(df_dwarfs[["teff_gspphot", "logg_gspphot", "radius_gspphot"]].corr())

    print("\n=== ТОП-20 ЭВОЛЮЦИОНИРОВАВШИХ ЗВЕЗД ПО РАДИУСУ ===")
    print(top_by_radius(df_evolved))
    save_frame(df_evolved, EVOLVED_SNAPSHOT_PATH)

    print("\n=== A/B/O REFERENCE LAYER ===")
    df_abo = load_abo_ref()
    print("Размер A/B/O reference:", df_abo.shape)
    print(df_abo.groupby("spec_class").size())
    print("\n=== A/B/O describe ===")
    print(df_abo.describe())

    abo_stats: pd.DataFrame = df_abo.groupby("spec_class")[
        ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
    ].agg(["count", "mean", "std", "min", "max"])
    print("\n=== A/B/O stats by class ===")
    print(abo_stats)

    save_frame(top_by_teff(df_abo), ABO_TOP20_PATH)

    plot_class_counts(
        df,
        title="Все MKGF: число объектов по классам",
        filename="all_class_counts.png",
    )
    plot_class_counts(
        df_dwarfs,
        title="Карлики (log g ≥ 4.0): число объектов по классам",
        filename="dwarfs_class_counts.png",
    )
    plot_class_counts(
        df_evolved,
        title="Эволюционировавшие (log g < 4.0): число объектов по классам",
        filename="evolved_class_counts.png",
    )
    plot_feature_histograms(
        df,
        filename_prefix="all",
        title_prefix="Все MKGF",
    )
    plot_feature_histograms(
        df_dwarfs,
        filename_prefix="dwarfs",
        title_prefix="Карлики",
    )
    plot_boxplots_dwarfs(df_dwarfs)
    plot_scatter_layers(df_dwarfs, df_evolved)
    plot_logg_radius_with_threshold(df)
    plot_correlation_heatmaps(df, df_dwarfs, df_evolved)

    print("\n=== DWARFS: mu и cov по классам M/K/G/F ===")
    for cls in ["M", "K", "G", "F"]:
        part = df_dwarfs[df_dwarfs["spec_class"] == cls]
        calc_gauss_stats(part, f"CLASS {cls}")

    print("\n=== DWARFS: mu и cov для подклассов M (Early/Mid/Late) ===")
    df_m = df_dwarfs[df_dwarfs["spec_class"] == "M"].copy()
    m_early = df_m[(df_m["teff_gspphot"] >= M_EARLY_MIN) & (df_m["teff_gspphot"] < M_EARLY_MAX)]
    m_mid = df_m[(df_m["teff_gspphot"] >= M_MID_MIN) & (df_m["teff_gspphot"] < M_EARLY_MIN)]
    m_late = df_m[df_m["teff_gspphot"] < M_MID_MIN]
    calc_gauss_stats(m_early, "M_EARLY [3500, 4000)")
    calc_gauss_stats(m_mid, "M_MID (3200-3500K)")
    calc_gauss_stats(m_late, "M_LATE (<3200K)")

    contrastive_view = os.getenv(CONTRASTIVE_VIEW_ENV, "").strip()
    if contrastive_view:
        print("\n=== CONTRASTIVE HOST/FIELD EDA ===")
        try:
            df_contrastive = load_contrastive_training_from_db(
                get_engine(),
                view_name=contrastive_view,
                population_col=CONTRASTIVE_POPULATION_COLUMN,
            )
            contrastive_stats = calc_contrastive_gauss_stats(
                df_contrastive,
                population_col=CONTRASTIVE_POPULATION_COLUMN,
                use_m_subclasses=True,
                shrink_alpha=CONTRASTIVE_SHRINK_ALPHA,
                min_population_size=CONTRASTIVE_MIN_POPULATION_SIZE,
            )
            CONTRASTIVE_GAUSS_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            contrastive_stats.to_csv(CONTRASTIVE_GAUSS_STATS_PATH, index=False)
            print_contrastive_gauss_stats(contrastive_stats)
            print("\nSaved contrastive readiness to", CONTRASTIVE_GAUSS_STATS_PATH)
        except Exception as exc:
            print(
                "\nContrastive host-field EDA skipped:",
                f"{contrastive_view} -> {exc}",
            )
    else:
        print(
            "\nCONTRASTIVE_HOST_FIELD_VIEW is not set. "
            "Skipping contrastive host-field EDA."
        )


if __name__ == "__main__":
    main()
