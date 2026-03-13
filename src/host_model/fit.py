"""Обучение legacy и контрастивных вариантов host-модели.

Модуль содержит два независимых контура обучения:

- legacy Gaussian model по host-популяции внутри MKGF dwarfs;
- contrastive `host-vs-field` model, которая является текущим боевым
  форматом для pipeline приоритизации.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from host_model.artifacts import (
    ContrastiveClassParams,
    ContrastiveGaussianModel,
    ContrastivePopulationParams,
    LegacyClassParams,
    LegacyGaussianModel,
)
from host_model.constants import (
    CONTRASTIVE_POPULATION_COLUMN,
    DWARF_CLASSES,
    FEATURES,
    LOGG_DWARF_MIN,
    M_EARLY_MAX,
    M_EARLY_MIN,
    M_LATE_MAX,
    M_MID_MAX,
    M_MID_MIN,
)
from host_model.gaussian_math import (
    cov_sample,
    shrink_covariance,
    stabilize_covariance,
    zscore_apply,
    zscore_fit,
)
from host_model.training_data import prepare_contrastive_training_df


def split_m_subclasses(df_m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Разбить M-dwarfs на ранние, средние и поздние подклассы по `Teff`."""
    teff = df_m["teff_gspphot"].astype(float)
    return {
        "M_EARLY": df_m[(teff >= M_EARLY_MIN) & (teff < M_EARLY_MAX)].copy(),
        "M_MID": df_m[(teff >= M_MID_MIN) & (teff < M_MID_MAX)].copy(),
        "M_LATE": df_m[teff < M_LATE_MAX].copy(),
    }


def build_contrastive_subsets(
    df_training: pd.DataFrame,
    population_col: str = CONTRASTIVE_POPULATION_COLUMN,
    use_m_subclasses: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Разбить contrastive training set на `host/field` подвыборки по label.

    Для класса `M` при включённом `use_m_subclasses` дополнительно
    формируются `M_EARLY`, `M_MID` и `M_LATE`.
    """
    subsets: dict[str, dict[str, pd.DataFrame]] = {}

    for spec_class in DWARF_CLASSES:
        class_df = df_training[df_training["spec_class"] == spec_class].copy()
        host_df = class_df[class_df[population_col]].copy()
        field_df = class_df[~class_df[population_col]].copy()

        if spec_class == "M" and use_m_subclasses:
            host_parts = split_m_subclasses(host_df)
            field_parts = split_m_subclasses(field_df)
            for label in ("M_EARLY", "M_MID", "M_LATE"):
                subsets[label] = {
                    "host": host_parts[label],
                    "field": field_parts[label],
                }
            continue

        subsets[spec_class] = {"host": host_df, "field": field_df}

    return subsets


def fit_population_gaussian(
    subset: pd.DataFrame,
    z_mu: np.ndarray,
    z_sigma: np.ndarray,
    shrink_alpha: float,
) -> ContrastivePopulationParams:
    """Оценить одну Gaussian-популяцию внутри contrastive-класса."""
    X = subset[FEATURES].astype(float).to_numpy()
    Xz = zscore_apply(X, z_mu, z_sigma)
    mu_z = Xz.mean(axis=0)
    cov_matrix = cov_sample(Xz)
    cov_matrix = shrink_covariance(cov_matrix, alpha=shrink_alpha)
    effective_cov, inv_cov, log_det_cov = stabilize_covariance(cov_matrix)
    return {
        "n": int(subset.shape[0]),
        "mu": mu_z.tolist(),
        "cov": cov_matrix.tolist(),
        "effective_cov": effective_cov.tolist(),
        "inv_cov": inv_cov.tolist(),
        "log_det_cov": float(log_det_cov),
    }


def fit_gaussian_model(
    df_dwarfs: pd.DataFrame,
    use_m_subclasses: bool = True,
    shrink_alpha: float = 0.15,
) -> LegacyGaussianModel:
    """Обучить legacy Gaussian model по MKGF dwarfs.

    Этот путь сохранён для обратной совместимости и диагностических
    сценариев. Текущий боевой pipeline использует contrastive-artifact.
    """
    required = ["spec_class", *FEATURES]
    missing = [col for col in required if col not in df_dwarfs.columns]
    if missing:
        raise ValueError(f"Missing required columns in df_dwarfs: {missing}")

    df = df_dwarfs.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No training rows remain after dropping NULL values.")

    X_all = df[FEATURES].astype(float).to_numpy()
    z_mu, z_sigma = zscore_fit(X_all)

    subsets: dict[str, pd.DataFrame] = {
        cls: df[df["spec_class"] == cls].copy() for cls in DWARF_CLASSES
    }
    if use_m_subclasses and "M" in subsets:
        subsets.pop("M", None)
        subsets.update(split_m_subclasses(df[df["spec_class"] == "M"].copy()))

    classes: dict[str, LegacyClassParams] = {}
    for label, subset in subsets.items():
        n = int(subset.shape[0])
        if n < 3:
            continue

        X = subset[FEATURES].astype(float).to_numpy()
        Xz = zscore_apply(X, z_mu, z_sigma)
        mu_z = Xz.mean(axis=0)
        cov_matrix = cov_sample(Xz)
        cov_matrix = shrink_covariance(cov_matrix, alpha=shrink_alpha)

        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            jitter = 1e-6
            inv_cov = np.linalg.inv(
                cov_matrix + jitter * np.eye(cov_matrix.shape[0])
            )

        classes[label] = cast(
            LegacyClassParams,
            {
                "n": n,
                "mu": mu_z.tolist(),
                "cov": cov_matrix.tolist(),
                "inv_cov": inv_cov.tolist(),
            },
        )

    if not classes:
        raise ValueError(
            "No Gaussian classes were fitted. "
            "Check input filters and class counts."
        )

    return {
        "global_mu": z_mu.tolist(),
        "global_sigma": z_sigma.tolist(),
        "classes": classes,
        "features": FEATURES,
        "meta": {
            "logg_dwarf_min": LOGG_DWARF_MIN,
            "use_m_subclasses": bool(use_m_subclasses),
            "shrink_alpha": float(shrink_alpha),
        },
    }


def fit_contrastive_gaussian_model(
    df_training: pd.DataFrame,
    population_col: str = CONTRASTIVE_POPULATION_COLUMN,
    use_m_subclasses: bool = True,
    shrink_alpha: float = 0.15,
    min_population_size: int = 2,
) -> ContrastiveGaussianModel:
    """Обучить contrastive `host-vs-field` модель внутри routed classes.

    Это основной training-path для текущего production artifact:
    по каждому class label оценивается пара Gaussian-популяций
    `host` и `field`, а в metadata фиксируются параметры scoring-контракта.
    """
    prepared = prepare_contrastive_training_df(
        df_training,
        population_col=population_col,
    )
    x_all = prepared[FEATURES].astype(float).to_numpy()
    z_mu, z_sigma = zscore_fit(x_all)

    classes: dict[str, ContrastiveClassParams] = {}
    subsets = build_contrastive_subsets(
        prepared,
        population_col=population_col,
        use_m_subclasses=use_m_subclasses,
    )
    for label, populations in subsets.items():
        host_subset = populations["host"]
        field_subset = populations["field"]
        if (
            int(host_subset.shape[0]) < min_population_size
            or int(field_subset.shape[0]) < min_population_size
        ):
            continue

        classes[label] = {
            "host": fit_population_gaussian(
                host_subset,
                z_mu=z_mu,
                z_sigma=z_sigma,
                shrink_alpha=shrink_alpha,
            ),
            "field": fit_population_gaussian(
                field_subset,
                z_mu=z_mu,
                z_sigma=z_sigma,
                shrink_alpha=shrink_alpha,
            ),
        }

    if not classes:
        raise ValueError(
            "No contrastive Gaussian classes were fitted. "
            "Check class/population counts after preprocessing."
        )

    return {
        "global_mu": z_mu.tolist(),
        "global_sigma": z_sigma.tolist(),
        "classes": classes,
        "features": FEATURES,
        "meta": {
            "model_version": "gaussian_host_field_v1",
            "score_mode": "host_vs_field_log_lr_v1",
            "population_col": str(population_col),
            "use_m_subclasses": bool(use_m_subclasses),
            "shrink_alpha": float(shrink_alpha),
            "min_population_size": int(min_population_size),
        },
    }


__all__ = [
    "build_contrastive_subsets",
    "fit_contrastive_gaussian_model",
    "fit_gaussian_model",
    "fit_population_gaussian",
    "split_m_subclasses",
]
