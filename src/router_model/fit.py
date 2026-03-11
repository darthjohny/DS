"""Обучение production-артефакта Gaussian router.

Модуль преобразует reference-выборку Gaia в сериализуемую модель:

- строит глобальную z-score нормализацию по `FEATURES`;
- формирует router labels из `spec_class` и `evolution_stage`;
- оценивает по каждому классу центр и ковариацию;
- сохраняет численно устойчивые поля для posterior-aware scoring.
"""

from __future__ import annotations

import pandas as pd

from router_model.artifacts import RouterClassParams, RouterModel
from router_model.db import FEATURES, ROUTER_VIEW
from router_model.labels import make_router_label, split_router_label
from router_model.math import (
    cov_sample,
    shrink_covariance,
    stabilize_covariance,
    zscore_apply,
    zscore_fit,
)

ROUTER_MODEL_VERSION = "gaussian_router_v1"


def fit_router_model(
    df_router: pd.DataFrame,
    shrink_alpha: float = 0.15,
    min_class_size: int = 3,
    source_view: str = ROUTER_VIEW,
) -> RouterModel:
    """Обучить Gaussian router по меткам `spec_class + evolution_stage`.

    Параметры
    ---------
    df_router
        Обучающая выборка с колонками `spec_class`, `evolution_stage`
        и признаками из `FEATURES`.
    shrink_alpha
        Коэффициент shrinkage для ковариационных матриц классов.
    min_class_size
        Минимальное число строк, необходимое для включения класса
        в итоговый artifact.
    source_view
        Имя relation, из которой была получена обучающая выборка.

    Возвращает
    ----------
    RouterModel
        Сериализуемый artifact c глобальной нормализацией, параметрами
        классов и metadata для production scoring.
    """
    required = ["spec_class", "evolution_stage"] + FEATURES
    missing = [col for col in required if col not in df_router.columns]
    if missing:
        raise ValueError(f"Missing required columns in df_router: {missing}")

    df = df_router.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No training rows remain after dropping NULL values.")

    df["router_label"] = [
        make_router_label(
            spec_class=spec_class,
            evolution_stage=evolution_stage,
        )
        for spec_class, evolution_stage in zip(
            df["spec_class"],
            df["evolution_stage"],
            strict=True,
        )
    ]

    x_all = df[FEATURES].astype(float).to_numpy()
    z_mu, z_sigma = zscore_fit(x_all)

    classes: dict[str, RouterClassParams] = {}
    for router_label, subset in df.groupby("router_label", sort=True):
        n = int(subset.shape[0])
        if n < min_class_size:
            continue

        x = subset[FEATURES].astype(float).to_numpy()
        xz = zscore_apply(x, z_mu, z_sigma)
        mu_z = xz.mean(axis=0)
        cov_matrix = cov_sample(xz)
        cov_matrix = shrink_covariance(cov_matrix, alpha=shrink_alpha)

        effective_cov, inv_cov, log_det_cov = stabilize_covariance(cov_matrix)

        router_label_str = str(router_label)
        spec_class, evolution_stage = split_router_label(router_label_str)
        classes[router_label_str] = {
            "n": n,
            "spec_class": spec_class,
            "evolution_stage": evolution_stage,
            "mu": mu_z.tolist(),
            "cov": cov_matrix.tolist(),
            "effective_cov": effective_cov.tolist(),
            "inv_cov": inv_cov.tolist(),
            "log_det_cov": float(log_det_cov),
        }

    if not classes:
        raise ValueError(
            "No router classes were fitted. "
            "Check class counts and training filters."
        )

    return {
        "global_mu": z_mu.tolist(),
        "global_sigma": z_sigma.tolist(),
        "classes": classes,
        "features": FEATURES,
        "meta": {
            "model_version": ROUTER_MODEL_VERSION,
            "source_view": source_view,
            "shrink_alpha": float(shrink_alpha),
            "min_class_size": int(min_class_size),
            "score_mode": "gaussian_log_posterior_v1",
            "prior_mode": "uniform",
        },
    }


__all__ = ["ROUTER_MODEL_VERSION", "fit_router_model"]
