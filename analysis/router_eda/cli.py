"""CLI-точка входа для router EDA."""

from __future__ import annotations

from analysis.router_eda.constants import (
    CLASS_COUNTS_PATH,
    GAUSS_STATS_PATH,
    PLOTS_DIR,
    SNAPSHOT_PATH,
)
from analysis.router_eda.data import (
    build_class_counts,
    build_router_training_query,
    ensure_router_labels,
    make_engine_from_env,
    read_sql_frame,
)
from analysis.router_eda.exports import save_router_artifacts
from analysis.router_eda.plots import (
    plot_router_correlation_heatmaps,
    plot_router_feature_boxplots,
    plot_router_feature_histograms,
    plot_router_label_counts,
    plot_router_scatter,
    set_default_theme,
)
from analysis.router_eda.readiness import calc_router_gauss_stats
from analysis.router_eda.stats import (
    print_basic_summary,
    print_class_counts,
    print_gaussian_readiness,
    print_null_report,
    print_router_describe,
)


def main() -> None:
    """Запустить полный router EDA сценарий end-to-end.

    Сценарий загружает reference-layer из БД, считает readiness-метрики,
    печатает статистические сводки и сохраняет табличные и графические
    артефакты в `data/eda/router`.
    """
    set_default_theme()
    engine = make_engine_from_env()
    df_router = ensure_router_labels(
        read_sql_frame(engine, build_router_training_query())
    )
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
    main()
