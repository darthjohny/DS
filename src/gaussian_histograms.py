"""
gaussian_histograms.py

Простая визуализация
для проверки гауссовой модели.

Что строим:
1) Гистограмма расстояния
   Махаланобиса (d_mahal)
2) Гистограмма similarity
3) Гистограмма d_mahal^2
   + теоретическая кривая chi-square(df=3)
4) Гистограммы d_mahal по классам M/K/G/F

Файлы сохраняются в data/plots/.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from model_gaussian import fit_gaussian_model
from model_gaussian import load_dwarfs_from_db
from model_gaussian import make_engine_from_env
from model_gaussian import score_df


PLOTS_DIR = "data/plots"


def ensure_plot_dir(path: str = PLOTS_DIR) -> None:
    """Создаёт папку для графиков."""
    os.makedirs(path, exist_ok=True)


def plot_hist_d_mahal(scored: pd.DataFrame, out_path: str) -> None:
    """Гистограмма d_mahal по всей выборке."""
    d = scored["d_mahal"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]

    plt.figure(figsize=(8, 5))
    plt.hist(d, bins=40, color="#4C72B0", alpha=0.85)
    plt.title("Распределение d_mahal (все объекты)")
    plt.xlabel("d_mahal")
    plt.ylabel("Количество")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_hist_similarity(scored: pd.DataFrame, out_path: str) -> None:
    """Гистограмма similarity по всей выборке."""
    s = scored["similarity"].to_numpy(dtype=float)
    s = s[np.isfinite(s)]

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=40, color="#55A868", alpha=0.85)
    plt.title("Распределение similarity (все объекты)")
    plt.xlabel("similarity")
    plt.ylabel("Количество")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_d2_vs_chi2(scored: pd.DataFrame, out_path: str) -> None:
    """Сравнение d_mahal^2
    с теоретическим chi-square(df=3).
    """
    d = scored["d_mahal"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    d2 = d**2

    if d2.size == 0:
        return

    x_max = float(np.percentile(d2, 99))
    x_max = max(x_max, 1.0)
    x = np.linspace(0.0, x_max, 300)
    y = chi2.pdf(x, df=3)

    plt.figure(figsize=(8, 5))
    plt.hist(
        d2,
        bins=50,
        density=True,
        alpha=0.65,
        color="#C44E52",
        label="Эмпирическое d_mahal^2",
    )
    plt.plot(x, y, color="black", linewidth=2, label="Chi-square(df=3)")
    plt.title(
        "Проверка формы распределения: d_mahal^2"
    )
    plt.xlabel("d_mahal^2")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_hist_d_mahal_by_class(scored: pd.DataFrame, out_path: str) -> None:
    """Гистограммы d_mahal по классам M/K/G/F."""
    classes: List[str] = ["M", "K", "G", "F"]
    colors = {
        "M": "#8172B2",
        "K": "#CCB974",
        "G": "#64B5CD",
        "F": "#E17C05",
    }

    plt.figure(figsize=(9, 6))
    for cls in classes:
        part = scored.loc[scored["spec_class"] == cls, "d_mahal"]
        d = part.to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        plt.hist(
            d,
            bins=30,
            density=True,
            alpha=0.35,
            label=cls,
            color=colors.get(cls, None),
        )

    plt.title("d_mahal по спектральным классам")
    plt.xlabel("d_mahal")
    plt.ylabel("Плотность")
    plt.legend(title="spec_class")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    """Запуск: обучение + скоринг
    + построение графиков.
    """
    print("=== Gaussian Histograms ===")

    ensure_plot_dir(PLOTS_DIR)

    engine = make_engine_from_env()
    df = load_dwarfs_from_db(engine)

    model = fit_gaussian_model(
        df_dwarfs=df,
        use_m_subclasses=True,
        shrink_alpha=0.15,
    )

    scored = score_df(
        model=model,
        df=df,
        spec_class_col="spec_class",
    )

    plot_hist_d_mahal(
        scored,
        os.path.join(PLOTS_DIR, "hist_d_mahal_all.png"),
    )
    plot_hist_similarity(
        scored,
        os.path.join(PLOTS_DIR, "hist_similarity_all.png"),
    )
    plot_d2_vs_chi2(
        scored,
        os.path.join(PLOTS_DIR, "hist_d2_vs_chi2.png"),
    )
    plot_hist_d_mahal_by_class(
        scored,
        os.path.join(PLOTS_DIR, "hist_d_mahal_by_class.png"),
    )

    print("Графики сохранены в:", PLOTS_DIR)
    print("- hist_d_mahal_all.png")
    print("- hist_similarity_all.png")
    print("- hist_d2_vs_chi2.png")
    print("- hist_d_mahal_by_class.png")


if __name__ == "__main__":
    main()
