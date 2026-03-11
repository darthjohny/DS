"""Posterior-aware метрики готовности для router EDA."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from analysis.router_eda.constants import (
    FEATURES,
    MIN_READY_CLASS_SIZE,
    ROUTER_PRIOR_MODE,
    ROUTER_SCORE_MODE,
    ROUTER_SHRINK_ALPHA,
    FloatArray,
)
from analysis.router_eda.data import feature_frame
from gaussian_router import (
    mahalanobis_distance,
    router_log_likelihood,
    shrink_covariance,
    stabilize_covariance,
    uniform_log_prior,
    zscore_apply,
    zscore_fit,
)


def is_missing_scalar(value: Any) -> bool:
    """Проверить, считается ли скаляр пропущенным значением."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def has_missing_values(values: Iterable[Any]) -> bool:
    """Проверить, есть ли пропуск хотя бы в одном признаке."""
    return any(is_missing_scalar(value) for value in values)


def calc_router_gauss_stats(df_router: pd.DataFrame) -> pd.DataFrame:
    """Посчитать Gaussian readiness-метрики для каждого `router_label`.

    Функция оценивает размер класса, ковариационные diagnostics и
    posterior-aware self-check метрики, которые помогают понять,
    готов ли данный router-класс к устойчивому scoring.
    """
    rows: list[dict[str, Any]] = []
    class_samples: dict[str, FloatArray] = {}
    class_params: dict[str, dict[str, Any]] = {}
    x_all: FloatArray = feature_frame(df_router).to_numpy(dtype=float)
    z_mu, z_sigma = zscore_fit(x_all)

    for router_label, subset in df_router.groupby("router_label", sort=True):
        x: FloatArray = feature_frame(subset).to_numpy(dtype=float)
        n = int(x.shape[0])
        base_row: dict[str, object] = {
            "router_label": str(router_label),
            "spec_class": str(subset["spec_class"].iloc[0]),
            "evolution_stage": str(subset["evolution_stage"].iloc[0]),
            "n_objects": n,
            "shrink_alpha": float(ROUTER_SHRINK_ALPHA),
            "score_mode": ROUTER_SCORE_MODE,
            "prior_mode": ROUTER_PRIOR_MODE,
        }

        if n < MIN_READY_CLASS_SIZE:
            rows.append(
                {
                    **base_row,
                    "mu_teff": float("nan"),
                    "mu_logg": float("nan"),
                    "mu_radius": float("nan"),
                    "det_cov": float("nan"),
                    "cond_cov": float("nan"),
                    "pd_cov": False,
                    "log_det_cov": float("nan"),
                    "cond_effective_cov": float("nan"),
                    "pd_effective_cov": False,
                    "n_competing_classes": 0,
                    "log_prior": float("nan"),
                    "self_log_posterior_mean": float("nan"),
                    "self_posterior_margin_mean": float("nan"),
                    "self_posterior_margin_min": float("nan"),
                    "self_posterior_win_rate": float("nan"),
                    "corr_teff_logg": float("nan"),
                    "corr_teff_radius": float("nan"),
                    "corr_logg_radius": float("nan"),
                }
            )
            continue

        mu: FloatArray = x.mean(axis=0)
        sigma: FloatArray = np.cov(x, rowvar=False, ddof=1)
        det_sigma = float(np.linalg.det(sigma))
        eigvals: FloatArray = np.linalg.eigvalsh(sigma)
        cond = float(np.linalg.cond(sigma))
        corr = feature_frame(subset).corr()
        xz = zscore_apply(x, z_mu, z_sigma)
        mu_z: FloatArray = xz.mean(axis=0)
        cov_z = np.cov(xz, rowvar=False, ddof=1)
        shrunk_cov = shrink_covariance(cov_z, alpha=ROUTER_SHRINK_ALPHA)
        effective_cov, inv_cov, log_det_cov = stabilize_covariance(shrunk_cov)
        effective_eigvals: FloatArray = np.linalg.eigvalsh(effective_cov)
        cond_effective_cov = float(np.linalg.cond(effective_cov))
        mu_any: Any = mu
        corr_any: Any = corr
        class_samples[str(router_label)] = xz
        class_params[str(router_label)] = {
            "mu": mu_z,
            "inv_cov": inv_cov,
            "log_det_cov": float(log_det_cov),
        }

        rows.append(
            {
                **base_row,
                "mu_teff": float(mu_any[0]),
                "mu_logg": float(mu_any[1]),
                "mu_radius": float(mu_any[2]),
                "det_cov": det_sigma,
                "cond_cov": cond,
                "pd_cov": bool(np.all(eigvals > 0.0)),
                "log_det_cov": float(log_det_cov),
                "cond_effective_cov": cond_effective_cov,
                "pd_effective_cov": bool(np.all(effective_eigvals > 0.0)),
                "n_competing_classes": 0,
                "log_prior": float("nan"),
                "self_log_posterior_mean": float("nan"),
                "self_posterior_margin_mean": float("nan"),
                "self_posterior_margin_min": float("nan"),
                "self_posterior_win_rate": float("nan"),
                "corr_teff_logg": float(corr_any.loc[FEATURES[0], FEATURES[1]]),
                "corr_teff_radius": float(corr_any.loc[FEATURES[0], FEATURES[2]]),
                "corr_logg_radius": float(corr_any.loc[FEATURES[1], FEATURES[2]]),
            }
        )

    if class_params:
        n_competing_classes = len(class_params)
        log_prior = uniform_log_prior(n_competing_classes)
        n_features = len(FEATURES)
        rows_by_label = {str(row["router_label"]): row for row in rows}

        for router_label, xz in class_samples.items():
            self_log_posteriors: list[float] = []
            self_margins: list[float] = []
            self_wins = 0

            for sample in xz:
                ranked: list[tuple[str, float]] = []
                self_log_posterior = float("nan")
                for other_label, params in class_params.items():
                    distance = mahalanobis_distance(
                        sample,
                        np.array(params["mu"], dtype=float),
                        np.array(params["inv_cov"], dtype=float),
                    )
                    log_likelihood = router_log_likelihood(
                        distance=distance,
                        log_det_cov=float(params["log_det_cov"]),
                        n_features=n_features,
                    )
                    log_posterior = float(log_likelihood + log_prior)
                    ranked.append((other_label, log_posterior))
                    if other_label == router_label:
                        self_log_posterior = log_posterior

                ranked.sort(key=lambda item: item[1], reverse=True)
                if ranked and ranked[0][0] == router_label:
                    self_wins += 1
                if len(ranked) > 1 and np.isfinite(self_log_posterior):
                    runner_up = next(
                        score
                        for label, score in ranked
                        if label != router_label
                    )
                    self_margins.append(self_log_posterior - runner_up)
                self_log_posteriors.append(self_log_posterior)

            row = rows_by_label[router_label]
            row["n_competing_classes"] = int(n_competing_classes)
            row["log_prior"] = float(log_prior)
            row["self_log_posterior_mean"] = float(np.mean(self_log_posteriors))
            row["self_posterior_margin_mean"] = float(np.mean(self_margins))
            row["self_posterior_margin_min"] = float(np.min(self_margins))
            row["self_posterior_win_rate"] = float(self_wins / len(xz))

    return pd.DataFrame(rows).sort_values("router_label", ignore_index=True)
