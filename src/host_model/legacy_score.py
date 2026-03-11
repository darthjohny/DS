"""Legacy scoring по расстоянию до Gaussian-центроидов host-популяции.

Этот модуль сохранён для обратной совместимости и диагностических
сценариев. Текущий боевой pipeline не использует его как основной
контур скоринга и вместо этого работает через contrastive `host-vs-field`
модель.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from host_model.artifacts import ScoreResult, require_legacy_scoring_model
from host_model.constants import FEATURES
from host_model.gaussian_math import (
    choose_m_subclass_label,
    has_missing_values,
    mahalanobis_distance,
    similarity_from_distance,
    zscore_apply,
)


def _empty_score(label: str) -> ScoreResult:
    """Вернуть нейтральный score payload для пустых или неподдерживаемых случаев."""
    return {
        "label": label,
        "d_mahal": float("nan"),
        "similarity": 0.0,
    }


def score_one(
    model: Mapping[str, Any],
    spec_class: str,
    teff: Any,
    logg: Any,
    radius: Any,
) -> ScoreResult:
    """Скорить одну звезду внутри заданного legacy класса `spec_class`."""
    require_legacy_scoring_model(model)
    if has_missing_values(teff, logg, radius):
        return _empty_score(str(spec_class))

    teff_val = float(teff)
    logg_val = float(logg)
    radius_val = float(radius)

    classes = model.get("classes", {})
    label = str(spec_class)
    if label == "M" and "M_EARLY" in classes:
        label = choose_m_subclass_label(teff_val)

    if label not in classes:
        return _empty_score(label)

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)
    x = np.array([teff_val, logg_val, radius_val], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    params = classes[label]
    mu = np.array(params["mu"], dtype=float)
    inv_cov = np.array(params["inv_cov"], dtype=float)
    d_mahal = mahalanobis_distance(xz, mu, inv_cov)

    return {
        "label": label,
        "d_mahal": float(d_mahal),
        "similarity": float(similarity_from_distance(d_mahal)),
    }


def score_one_all_classes(
    model: Mapping[str, Any],
    teff: Any,
    logg: Any,
    radius: Any,
) -> ScoreResult:
    """Скорить одну звезду по всем legacy Gaussian classes и взять лучший матч."""
    require_legacy_scoring_model(model)
    if has_missing_values(teff, logg, radius):
        return _empty_score("UNKNOWN")

    classes = model.get("classes", {})
    if not classes:
        return _empty_score("UNKNOWN")

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)
    x = np.array([float(teff), float(logg), float(radius)], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    best_label = "UNKNOWN"
    best_d = float("inf")
    for label, params in classes.items():
        mu = np.array(params["mu"], dtype=float)
        inv_cov = np.array(params["inv_cov"], dtype=float)
        distance = mahalanobis_distance(xz, mu, inv_cov)
        if distance < best_d:
            best_d = distance
            best_label = label

    if best_label == "UNKNOWN":
        return _empty_score("UNKNOWN")

    return {
        "label": best_label,
        "d_mahal": float(best_d),
        "similarity": float(similarity_from_distance(best_d)),
    }


def score_df(
    model: Mapping[str, Any],
    df: pd.DataFrame,
    spec_class_col: str = "spec_class",
) -> pd.DataFrame:
    """Скорить DataFrame через legacy path, используя колонку `spec_class`."""
    require_legacy_scoring_model(model)
    required = [spec_class_col, *FEATURES]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in df: {col}")

    rows: list[ScoreResult] = []
    for _, row in df.iterrows():
        rows.append(
            score_one(
                model=model,
                spec_class=str(row[spec_class_col]),
                teff=row["teff_gspphot"],
                logg=row["logg_gspphot"],
                radius=row["radius_gspphot"],
            )
        )

    result = df.copy()
    result["gauss_label"] = [item["label"] for item in rows]
    result["d_mahal"] = [item["d_mahal"] for item in rows]
    result["similarity"] = [item["similarity"] for item in rows]
    return result


def score_df_all_classes(
    model: Mapping[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Скорить DataFrame по всем legacy Gaussian classes без fixed `spec_class`."""
    require_legacy_scoring_model(model)
    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required column in df: {col}")

    rows: list[ScoreResult] = []
    for _, row in df.iterrows():
        rows.append(
            score_one_all_classes(
                model=model,
                teff=row["teff_gspphot"],
                logg=row["logg_gspphot"],
                radius=row["radius_gspphot"],
            )
        )

    result = df.copy()
    result["gauss_label"] = [item["label"] for item in rows]
    result["d_mahal"] = [item["d_mahal"] for item in rows]
    result["similarity"] = [item["similarity"] for item in rows]
    return result


__all__ = [
    "score_df",
    "score_df_all_classes",
    "score_one",
    "score_one_all_classes",
]
