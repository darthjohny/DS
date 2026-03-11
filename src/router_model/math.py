"""Численные примитивы Gaussian router.

Здесь собраны низкоуровневые операции, которые используются и при
обучении, и при scoring: z-score нормализация, ковариации, shrinkage,
Mahalanobis distance и элементы Gaussian posterior-контракта.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


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
    """Применить z-score нормализацию с заранее оценёнными параметрами."""
    return (X - mu) / sigma


def cov_sample(X: np.ndarray) -> np.ndarray:
    """Оценить выборочную ковариацию по матрице признаков."""
    if X.shape[0] < 2:
        raise ValueError(
            "At least 2 rows are required to estimate covariance."
        )
    return np.cov(X, rowvar=False, ddof=1)


def shrink_covariance(cov_matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Применить shrinkage ковариации в сторону диагональной матрицы."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    diag = np.diag(np.diag(cov_matrix))
    return (1.0 - alpha) * cov_matrix + alpha * diag


def mahalanobis_distance(
    x: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
) -> float:
    """Вычислить расстояние Махаланобиса до центра класса."""
    delta = x - mu
    value = float(delta.T @ inv_cov @ delta)
    return math.sqrt(max(value, 0.0))


def similarity_from_distance(distance: float) -> float:
    """Преобразовать расстояние Махаланобиса в bounded similarity."""
    return 1.0 / (1.0 + float(distance))


def stabilize_covariance(
    cov_matrix: np.ndarray,
    jitter: float = 1e-6,
    max_attempts: int = 8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Стабилизировать ковариацию и вернуть её рабочий численный вариант.

    Возвращает эффективную ковариацию, обратную матрицу и `log(det)`.
    Если исходная матрица вырожденная или плохо обусловленная, функция
    постепенно добавляет `jitter` к диагонали.
    """
    effective_cov = np.array(cov_matrix, dtype=float, copy=True)
    base_eye = np.eye(effective_cov.shape[0], dtype=float)
    current_jitter = float(jitter)

    for _ in range(max_attempts):
        sign, log_det = np.linalg.slogdet(effective_cov)
        if sign > 0.0:
            try:
                inv_cov = np.linalg.inv(effective_cov)
                return effective_cov, inv_cov, float(log_det)
            except np.linalg.LinAlgError:
                pass
        effective_cov = effective_cov + current_jitter * base_eye
        current_jitter *= 10.0

    raise ValueError("Failed to stabilize covariance matrix for router.")


def router_log_likelihood(
    distance: float,
    log_det_cov: float,
    n_features: int,
) -> float:
    """Вычислить полный Gaussian log-likelihood для одного router-класса."""
    distance_sq = float(distance) * float(distance)
    norm_term = float(n_features) * math.log(2.0 * math.pi)
    return -0.5 * (norm_term + float(log_det_cov) + distance_sq)


def uniform_log_prior(n_classes: int) -> float:
    """Вычислить равномерный log-prior по числу доступных классов."""
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")
    return -math.log(float(n_classes))


__all__ = [
    "EPS",
    "cov_sample",
    "mahalanobis_distance",
    "router_log_likelihood",
    "shrink_covariance",
    "similarity_from_distance",
    "stabilize_covariance",
    "uniform_log_prior",
    "zscore_apply",
    "zscore_fit",
]
