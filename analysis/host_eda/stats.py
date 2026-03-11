"""Статистические сводки для host EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.host_eda.constants import FloatArray
from analysis.host_eda.data import feature_frame


def calc_gauss_stats(df_part: pd.DataFrame, label: str) -> None:
    """Напечатать `mu/cov` и QA-метрики для одного среза популяции."""
    x: FloatArray = feature_frame(df_part).to_numpy(dtype=float)
    n = x.shape[0]

    if n < 5:
        print(f"\n[{label}] Слишком мало объектов для устойчивой cov: n={n}")
        return

    mu: FloatArray = x.mean(axis=0)
    sigma: FloatArray = np.cov(x, rowvar=False, ddof=1)

    det_sigma = float(np.linalg.det(sigma))
    eigvals: FloatArray = np.linalg.eigvalsh(sigma)
    cond = float(np.linalg.cond(sigma))

    print(f"\n[{label}] n={n}")
    print("mu =", mu)
    print("cov =\n", sigma)
    print("det(cov) =", det_sigma)
    print("eigenvalues(cov) =", eigvals)
    print("cond(cov) =", cond)
    print("PD (все eigenvalues > 0):", bool(np.all(eigvals > 0)))


def build_group_stats(df_part: pd.DataFrame) -> pd.DataFrame:
    """Собрать агрегаты `mean/std/min/max` по спектральным классам."""
    return df_part.groupby("spec_class")[list(feature_frame(df_part).columns)].agg(
        ["mean", "std", "min", "max"]
    )


def top_by_radius(df_part: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Вернуть объекты с наибольшим радиусом."""
    return df_part.sort_values("radius_gspphot", ascending=False).head(limit)


def top_by_teff(df_part: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Вернуть объекты с наибольшей эффективной температурой."""
    return df_part.sort_values("teff_gspphot", ascending=False).head(limit)
