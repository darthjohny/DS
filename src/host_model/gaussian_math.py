"""Численные helpers, общие для обучения и scoring host-модели.

Модуль содержит базовые операции для обеих веток host-модели:

- z-score нормализацию;
- ковариации и shrinkage;
- Mahalanobis distance и Gaussian log-likelihood;
- выбор M-подкласса;
- вычисление contrastive `P(host | x, class)`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from host_model.artifacts import ContrastivePopulationParams
from host_model.constants import (
    EPS,
    M_EARLY_MAX,
    M_EARLY_MIN,
    M_LATE_MAX,
    M_MID_MAX,
    M_MID_MIN,
)


def zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Оценить параметры глобальной z-score нормализации."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(np.abs(sigma) < EPS, EPS, sigma)
    return mu, sigma


def zscore_apply(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Применить z-score нормализацию с уже известными параметрами."""
    return (X - mu) / sigma


def cov_sample(X: np.ndarray) -> np.ndarray:
    """Оценить выборочную ковариацию по матрице признаков."""
    if X.shape[0] < 2:
        raise ValueError(
            "At least 2 rows are required to estimate covariance."
        )
    return np.cov(X, rowvar=False, ddof=1)


def shrink_covariance(cov_matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Применить shrinkage ковариации в сторону диагонали."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    diag = np.diag(np.diag(cov_matrix))
    return (1.0 - alpha) * cov_matrix + alpha * diag


def stabilize_covariance(
    cov_matrix: np.ndarray,
    jitter: float = 1e-6,
    max_attempts: int = 8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Стабилизировать ковариацию и вернуть рабочий численный payload.

    Возвращает эффективную ковариацию, обратную матрицу и `log(det)`.
    Если исходная матрица плохо обусловлена, функция добавляет `jitter`
    к диагонали до получения устойчивого результата.
    """
    effective_cov = np.array(cov_matrix, dtype=float, copy=True)
    identity = np.eye(effective_cov.shape[0], dtype=float)
    current_jitter = float(jitter)

    for _ in range(max_attempts):
        sign, log_det_cov = np.linalg.slogdet(effective_cov)
        if sign > 0.0:
            try:
                inv_cov = np.linalg.inv(effective_cov)
                return effective_cov, inv_cov, float(log_det_cov)
            except np.linalg.LinAlgError:
                pass
        effective_cov = effective_cov + current_jitter * identity
        current_jitter *= 10.0

    raise ValueError("Failed to stabilize covariance matrix for host model.")


def mahalanobis_distance(
    x: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
) -> float:
    """Вычислить расстояние Махаланобиса до Gaussian-центра."""
    delta = x - mu
    value = float(delta.T @ inv_cov @ delta)
    return math.sqrt(max(value, 0.0))


def gaussian_log_likelihood(
    x: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    log_det_cov: float,
) -> float:
    """Вычислить полный Gaussian log-likelihood для стандартизованного вектора."""
    delta = x - mu
    quad_form = float(delta.T @ inv_cov @ delta)
    ndim = float(x.shape[0])
    return float(
        -0.5 * quad_form
        - 0.5 * float(log_det_cov)
        - 0.5 * ndim * math.log(2.0 * math.pi)
    )


def similarity_from_distance(d: float) -> float:
    """Преобразовать расстояние Махаланобиса в bounded similarity."""
    return 1.0 / (1.0 + float(d))


def choose_m_subclass_label(teff: float) -> str:
    """Определить label M-подкласса по эффективной температуре."""
    if M_EARLY_MIN <= teff < M_EARLY_MAX:
        return "M_EARLY"
    if M_MID_MIN <= teff < M_MID_MAX:
        return "M_MID"
    if teff < M_LATE_MAX:
        return "M_LATE"
    return "M_UNKNOWN"


def is_missing_scalar(value: Any) -> bool:
    """Проверить, считается ли скаляр пропущенным значением."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def has_missing_values(teff: Any, logg: Any, radius: Any) -> bool:
    """Проверить, что среди базовых признаков нет пропусков."""
    return (
        is_missing_scalar(teff)
        or is_missing_scalar(logg)
        or is_missing_scalar(radius)
    )


def contrastive_host_posterior(
    host_log_likelihood: float,
    field_log_likelihood: float,
) -> float:
    """Вычислить `P(host | x, class)` при равных priors `host/field`."""
    max_log_score = max(host_log_likelihood, field_log_likelihood)
    log_denom = max_log_score + math.log(
        math.exp(host_log_likelihood - max_log_score)
        + math.exp(field_log_likelihood - max_log_score)
    )
    return float(math.exp(host_log_likelihood - log_denom))


def population_log_likelihood(
    xz: np.ndarray,
    params: ContrastivePopulationParams,
) -> float:
    """Скорить один вектор против одной fitted `host` или `field` Gaussian."""
    mu = np.array(params["mu"], dtype=float)
    inv_cov = np.array(params["inv_cov"], dtype=float)
    log_det_cov = float(params["log_det_cov"])
    return gaussian_log_likelihood(
        x=xz,
        mu=mu,
        inv_cov=inv_cov,
        log_det_cov=log_det_cov,
    )


__all__ = [
    "choose_m_subclass_label",
    "contrastive_host_posterior",
    "cov_sample",
    "gaussian_log_likelihood",
    "has_missing_values",
    "is_missing_scalar",
    "mahalanobis_distance",
    "population_log_likelihood",
    "shrink_covariance",
    "similarity_from_distance",
    "stabilize_covariance",
    "zscore_apply",
    "zscore_fit",
]
