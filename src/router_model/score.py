"""Инференс и расчёт score для Gaussian router.

Модуль реализует posterior-aware scoring поверх production artifact:

- подготавливает ковариационные параметры класса;
- считает расстояние, log-likelihood и log-posterior по всем классам;
- выбирает победителя по `router_log_posterior`, а не только по
  минимальному Mahalanobis distance;
- собирает табличный результат для боевого pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from router_model.artifacts import (
    RouterClassParams,
    RouterModel,
    RouterScoreResult,
)
from router_model.db import FEATURES
from router_model.labels import has_missing_values, split_router_label
from router_model.math import (
    mahalanobis_distance,
    router_log_likelihood,
    similarity_from_distance,
    stabilize_covariance,
    uniform_log_prior,
    zscore_apply,
)
from router_model.ood import apply_ood_policy, build_unknown_router_score


def unpack_router_covariance(
    params: RouterClassParams,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Извлечь ковариационный payload класса с fallback для legacy-artifact.

    Если в artifact уже есть `effective_cov` и `log_det_cov`, функция
    использует их напрямую. Для старого формата она восстанавливает
    рабочие поля из `cov` и при необходимости стабилизирует матрицу.
    """
    inv_cov = np.array(params["inv_cov"], dtype=float)

    if "effective_cov" in params and "log_det_cov" in params:
        effective_cov = np.array(params["effective_cov"], dtype=float)
        log_det_cov = float(params["log_det_cov"])
        return effective_cov, inv_cov, log_det_cov

    cov_matrix = np.array(params["cov"], dtype=float)
    sign, log_det_cov = np.linalg.slogdet(cov_matrix)
    if sign > 0.0:
        return cov_matrix, inv_cov, float(log_det_cov)

    return stabilize_covariance(cov_matrix)


RouterRankedCandidate = tuple[str, float, float, float]


def rank_router_candidates(
    model: RouterModel,
    xz: np.ndarray,
) -> list[RouterRankedCandidate]:
    """Посчитать raw ranking по всем router-классам для одной строки."""
    classes = model.get("classes", {})
    if not classes:
        return []

    ranked: list[RouterRankedCandidate] = []
    log_prior = uniform_log_prior(len(classes))
    n_features = len(FEATURES)
    for router_label, params in classes.items():
        mu = np.array(params["mu"], dtype=float)
        _, inv_cov, log_det_cov = unpack_router_covariance(params)
        distance = mahalanobis_distance(xz, mu, inv_cov)
        log_likelihood = router_log_likelihood(
            distance=distance,
            log_det_cov=log_det_cov,
            n_features=n_features,
        )
        ranked.append(
            (
                router_label,
                distance,
                log_likelihood,
                log_likelihood + log_prior,
            )
        )
    return ranked


def build_raw_router_score(
    ranked: list[RouterRankedCandidate],
    model_version: str,
) -> RouterScoreResult:
    """Собрать raw router result до применения OOD policy."""
    if not ranked:
        return build_unknown_router_score(model_version=model_version)

    ranked_by_distance = sorted(ranked, key=lambda item: item[1])
    ranked_by_posterior = sorted(ranked, key=lambda item: item[3], reverse=True)

    (
        best_label,
        best_distance,
        best_log_likelihood,
        best_log_posterior,
    ) = ranked_by_posterior[0]
    second_label = ranked_by_posterior[1][0] if len(ranked) > 1 else best_label
    margin = float("nan")
    posterior_margin = float("nan")
    if len(ranked) > 1:
        margin = float(ranked_by_distance[1][1] - ranked_by_distance[0][1])
        posterior_margin = float(
            ranked_by_posterior[0][3] - ranked_by_posterior[1][3]
        )

    d_mahal_router = float(best_distance)
    router_similarity = float(similarity_from_distance(best_distance))
    router_log_likelihood = float(best_log_likelihood)
    router_log_posterior = float(best_log_posterior)
    try:
        spec_class, evolution_stage = split_router_label(best_label)
    except ValueError:
        return build_unknown_router_score(
            model_version=model_version,
            diagnostics={
                "d_mahal_router": d_mahal_router,
                "router_similarity": router_similarity,
                "router_log_likelihood": router_log_likelihood,
                "router_log_posterior": router_log_posterior,
                "second_best_label": second_label,
                "margin": margin,
                "posterior_margin": posterior_margin,
            },
        )

    return {
        "predicted_spec_class": spec_class,
        "predicted_evolution_stage": evolution_stage,
        "router_label": best_label,
        "d_mahal_router": d_mahal_router,
        "router_similarity": router_similarity,
        "router_log_likelihood": router_log_likelihood,
        "router_log_posterior": router_log_posterior,
        "second_best_label": second_label,
        "margin": margin,
        "posterior_margin": posterior_margin,
        "model_version": model_version,
    }


def score_router_one(
    model: RouterModel,
    teff: Any,
    logg: Any,
    radius: Any,
) -> RouterScoreResult:
    """Скорить одну звезду относительно всех router-классов.

    Функция нормализует признаки через глобальные параметры модели,
    считает метрики по всем Gaussian-компонентам и выбирает победителя
    по `router_log_posterior`. В результат также входят diagnostics:
    второе место, distance-margin и posterior-margin.
    """
    model_version = model["meta"]["model_version"]
    if has_missing_values([teff, logg, radius]):
        return build_unknown_router_score(model_version=model_version)

    classes = model.get("classes", {})
    if not classes:
        return build_unknown_router_score(model_version=model_version)

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)
    x = np.array([float(teff), float(logg), float(radius)], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)
    ranked = rank_router_candidates(model=model, xz=xz)
    raw_result = build_raw_router_score(ranked=ranked, model_version=model_version)
    return apply_ood_policy(result=raw_result, meta=model["meta"])


def score_router_df(
    model: RouterModel,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Скорить DataFrame относительно router-модели.

    Требует наличие всех признаков из `FEATURES` и добавляет в копию
    входного DataFrame поля router inference:
    `predicted_spec_class`, `predicted_evolution_stage`, `router_label`,
    `d_mahal_router`, `router_similarity`, `router_log_likelihood`,
    `router_log_posterior`, `second_best_label`, `margin`,
    `posterior_margin`, `model_version`.
    """
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    rows: list[RouterScoreResult] = []
    for teff, logg, radius in df[FEATURES].itertuples(index=False, name=None):
        rows.append(
            score_router_one(
                model=model,
                teff=teff,
                logg=logg,
                radius=radius,
            )
        )

    result = df.copy()
    result["predicted_spec_class"] = [
        item["predicted_spec_class"] for item in rows
    ]
    result["predicted_evolution_stage"] = [
        item["predicted_evolution_stage"] for item in rows
    ]
    result["router_label"] = [item["router_label"] for item in rows]
    result["d_mahal_router"] = [item["d_mahal_router"] for item in rows]
    result["router_similarity"] = [item["router_similarity"] for item in rows]
    result["router_log_likelihood"] = [
        item["router_log_likelihood"] for item in rows
    ]
    result["router_log_posterior"] = [
        item["router_log_posterior"] for item in rows
    ]
    result["second_best_label"] = [item["second_best_label"] for item in rows]
    result["margin"] = [item["margin"] for item in rows]
    result["posterior_margin"] = [item["posterior_margin"] for item in rows]
    result["model_version"] = [item["model_version"] for item in rows]
    return result


__all__ = [
    "build_raw_router_score",
    "rank_router_candidates",
    "score_router_df",
    "score_router_one",
    "unpack_router_covariance",
]
