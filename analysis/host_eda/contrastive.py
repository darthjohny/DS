"""Контрастивный анализ `host vs field` для host EDA."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from analysis.host_eda.constants import (
    CONTRASTIVE_MIN_POPULATION_SIZE,
    CONTRASTIVE_MODEL_VERSION,
    CONTRASTIVE_SCORE_MODE,
    CONTRASTIVE_SHRINK_ALPHA,
)
from analysis.host_eda.data import feature_frame
from model_gaussian import (
    CONTRASTIVE_POPULATION_COLUMN,
    build_contrastive_subsets,
    contrastive_host_posterior,
    fit_population_gaussian,
    gaussian_log_likelihood,
    prepare_contrastive_training_df,
    zscore_apply,
    zscore_fit,
)


def calc_contrastive_gauss_stats(
    df_part: pd.DataFrame,
    population_col: str = CONTRASTIVE_POPULATION_COLUMN,
    use_m_subclasses: bool = True,
    shrink_alpha: float = CONTRASTIVE_SHRINK_ALPHA,
    min_population_size: int = CONTRASTIVE_MIN_POPULATION_SIZE,
) -> pd.DataFrame:
    """Посчитать readiness и separability для `host-vs-field` scoring.

    Функция использует контракты и численные helpers из `model_gaussian`,
    чтобы оценить, насколько устойчиво разделяются популяции `host` и
    `field` внутри каждого class label.
    """
    prepared = prepare_contrastive_training_df(
        df_part,
        population_col=population_col,
    )
    x_all = feature_frame(prepared).to_numpy(dtype=float)
    z_mu, z_sigma = zscore_fit(x_all)

    rows: list[dict[str, Any]] = []
    subsets = build_contrastive_subsets(
        prepared,
        population_col=population_col,
        use_m_subclasses=use_m_subclasses,
    )

    for label, populations in subsets.items():
        host_subset = populations["host"].copy()
        field_subset = populations["field"].copy()
        host_n = int(host_subset.shape[0])
        field_n = int(field_subset.shape[0])
        ready = host_n >= min_population_size and field_n >= min_population_size

        row: dict[str, Any] = {
            "gauss_label": label,
            "host_n": host_n,
            "field_n": field_n,
            "total_n": host_n + field_n,
            "population_col": str(population_col),
            "use_m_subclasses": bool(use_m_subclasses),
            "shrink_alpha": float(shrink_alpha),
            "model_version": CONTRASTIVE_MODEL_VERSION,
            "score_mode": CONTRASTIVE_SCORE_MODE,
            "ready_for_contrastive_model": bool(ready),
        }

        if not ready:
            rows.append(row)
            continue

        host_params = fit_population_gaussian(
            host_subset,
            z_mu=z_mu,
            z_sigma=z_sigma,
            shrink_alpha=shrink_alpha,
        )
        field_params = fit_population_gaussian(
            field_subset,
            z_mu=z_mu,
            z_sigma=z_sigma,
            shrink_alpha=shrink_alpha,
        )

        host_cov = np.array(host_params["effective_cov"], dtype=float)
        field_cov = np.array(field_params["effective_cov"], dtype=float)
        host_eigvals = np.linalg.eigvalsh(host_cov)
        field_eigvals = np.linalg.eigvalsh(field_cov)

        labeled_part = pd.concat([host_subset, field_subset], ignore_index=True)
        xz = zscore_apply(
            labeled_part[feature_frame(labeled_part).columns].to_numpy(dtype=float),
            z_mu,
            z_sigma,
        )

        host_mu = np.array(host_params["mu"], dtype=float)
        host_inv_cov = np.array(host_params["inv_cov"], dtype=float)
        host_log_det_cov = float(host_params["log_det_cov"])
        field_mu = np.array(field_params["mu"], dtype=float)
        field_inv_cov = np.array(field_params["inv_cov"], dtype=float)
        field_log_det_cov = float(field_params["log_det_cov"])

        host_ll = np.array(
            [
                gaussian_log_likelihood(
                    x=row_x,
                    mu=host_mu,
                    inv_cov=host_inv_cov,
                    log_det_cov=host_log_det_cov,
                )
                for row_x in xz
            ],
            dtype=float,
        )
        field_ll = np.array(
            [
                gaussian_log_likelihood(
                    x=row_x,
                    mu=field_mu,
                    inv_cov=field_inv_cov,
                    log_det_cov=field_log_det_cov,
                )
                for row_x in xz
            ],
            dtype=float,
        )
        host_log_lr = host_ll - field_ll
        host_posterior = np.array(
            [
                contrastive_host_posterior(
                    host_log_likelihood=float(host_score),
                    field_log_likelihood=float(field_score),
                )
                for host_score, field_score in zip(host_ll, field_ll, strict=True)
            ],
            dtype=float,
        )

        is_host: npt.NDArray[np.bool_] = labeled_part[population_col].to_numpy(dtype=bool)
        predicted_host: npt.NDArray[np.bool_] = np.greater_equal(host_ll, field_ll)
        contrastive_accuracy = float(np.count_nonzero(predicted_host == is_host)) / float(
            is_host.size
        )

        row.update(
            {
                "host_log_det_cov": host_log_det_cov,
                "field_log_det_cov": field_log_det_cov,
                "host_cond_cov": float(np.linalg.cond(host_cov)),
                "field_cond_cov": float(np.linalg.cond(field_cov)),
                "host_pd_cov": bool(np.all(host_eigvals > 0.0)),
                "field_pd_cov": bool(np.all(field_eigvals > 0.0)),
                "host_log_lr_mean_host": float(np.mean(host_log_lr[is_host])),
                "host_log_lr_mean_field": float(np.mean(host_log_lr[~is_host])),
                "host_posterior_mean_host": float(np.mean(host_posterior[is_host])),
                "host_posterior_mean_field": float(np.mean(host_posterior[~is_host])),
                "contrastive_margin_mean": float(np.mean(np.abs(host_log_lr))),
                "contrastive_accuracy": contrastive_accuracy,
                "centroid_distance_z": float(np.linalg.norm(host_mu - field_mu)),
            }
        )
        rows.append(row)

    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    return stats.sort_values("gauss_label").reset_index(drop=True)


def print_contrastive_gauss_stats(stats: pd.DataFrame) -> None:
    """Напечатать компактную таблицу readiness для contrastive host EDA."""
    if stats.empty:
        print("\n=== CONTRASTIVE HOST/FIELD: данных для readiness нет ===")
        return

    preview_cols = [
        "gauss_label",
        "host_n",
        "field_n",
        "ready_for_contrastive_model",
        "host_log_det_cov",
        "field_log_det_cov",
        "contrastive_accuracy",
        "host_log_lr_mean_host",
        "host_log_lr_mean_field",
    ]
    available_cols = [col for col in preview_cols if col in stats.columns]
    print("\n=== CONTRASTIVE HOST/FIELD READINESS ===")
    print(stats[available_cols].to_string(index=False))
