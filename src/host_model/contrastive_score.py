"""Контрастивный `host-vs-field` scoring внутри routed stellar classes.

Модуль реализует текущий production scoring-path host-модели:

- выбирает корректный class label, включая M-subclasses;
- считает log-likelihood для `host` и `field` популяций;
- возвращает diagnostics `host_log_lr` и bounded `host_posterior`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from host_model.artifacts import (
    ContrastiveClassParams,
    ContrastiveScoreResult,
    require_contrastive_scoring_model,
)
from host_model.constants import FEATURES
from host_model.gaussian_math import (
    choose_m_subclass_label,
    contrastive_host_posterior,
    has_missing_values,
    population_log_likelihood,
    zscore_apply,
)


def _empty_contrastive_score(label: str) -> ContrastiveScoreResult:
    """Вернуть нейтральный результат для пустого contrastive score."""
    return {
        "label": label,
        "host_log_likelihood": float("nan"),
        "field_log_likelihood": float("nan"),
        "host_log_lr": float("nan"),
        "host_posterior": 0.0,
    }


def resolve_contrastive_label(
    model: Mapping[str, Any],
    spec_class: str,
    teff: float,
) -> str:
    """Определить label класса host-модели, включая M-подклассы."""
    classes = model.get("classes", {})
    label = str(spec_class)
    if label == "M" and "M_EARLY" in classes:
        return choose_m_subclass_label(teff)
    return label


def score_one_contrastive(
    model: Mapping[str, Any],
    spec_class: str,
    teff: Any,
    logg: Any,
    radius: Any,
) -> ContrastiveScoreResult:
    """Скорить одну звезду по `host` и `field` Gaussian внутри одного класса."""
    require_contrastive_scoring_model(model)
    if has_missing_values(teff, logg, radius):
        return _empty_contrastive_score(str(spec_class))

    teff_val = float(teff)
    logg_val = float(logg)
    radius_val = float(radius)

    label = resolve_contrastive_label(model, spec_class=str(spec_class), teff=teff_val)
    classes = model.get("classes", {})
    if label not in classes:
        return _empty_contrastive_score(label)

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)
    x = np.array([teff_val, logg_val, radius_val], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    params = cast(ContrastiveClassParams, classes[label])
    host_log_likelihood = population_log_likelihood(xz, params["host"])
    field_log_likelihood = population_log_likelihood(xz, params["field"])
    host_log_lr = float(host_log_likelihood - field_log_likelihood)

    return {
        "label": label,
        "host_log_likelihood": float(host_log_likelihood),
        "field_log_likelihood": float(field_log_likelihood),
        "host_log_lr": host_log_lr,
        "host_posterior": contrastive_host_posterior(
            host_log_likelihood=host_log_likelihood,
            field_log_likelihood=field_log_likelihood,
        ),
    }


def score_df_contrastive(
    model: Mapping[str, Any],
    df: pd.DataFrame,
    spec_class_col: str = "spec_class",
) -> pd.DataFrame:
    """Скорить DataFrame текущей contrastive host-моделью.

    В копию входного DataFrame добавляются поля:
    `gauss_label`, `host_log_likelihood`, `field_log_likelihood`,
    `host_log_lr`, `host_posterior`.
    """
    require_contrastive_scoring_model(model)
    required = [spec_class_col, *FEATURES]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in df: {col}")

    rows: list[ContrastiveScoreResult] = []
    for _, row in df.iterrows():
        rows.append(
            score_one_contrastive(
                model=model,
                spec_class=str(row[spec_class_col]),
                teff=row["teff_gspphot"],
                logg=row["logg_gspphot"],
                radius=row["radius_gspphot"],
            )
        )

    result = df.copy()
    result["gauss_label"] = [item["label"] for item in rows]
    result["host_log_likelihood"] = [
        item["host_log_likelihood"] for item in rows
    ]
    result["field_log_likelihood"] = [
        item["field_log_likelihood"] for item in rows
    ]
    result["host_log_lr"] = [item["host_log_lr"] for item in rows]
    result["host_posterior"] = [item["host_posterior"] for item in rows]
    return result


__all__ = [
    "resolve_contrastive_label",
    "score_df_contrastive",
    "score_one_contrastive",
]
