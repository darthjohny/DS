"""Тесты для Gaussian router."""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaussian_router import (
    fit_router_model,
    load_router_model,
    make_router_label,
    save_router_model,
    score_router_df,
    split_router_label,
)

ROUTER_TRAIN_COLUMNS = [
    "source_id",
    "spec_class",
    "evolution_stage",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
]

ROUTER_SCORE_COLUMNS = [
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
]

RouterTrainRow = tuple[int, str, str, float, float, float]
RouterScoreRow = tuple[float, float, float]


def scalar_to_float(value: object) -> float:
    """Преобразовать pandas-скаляр во `float` с явной runtime-проверкой."""
    if isinstance(value, (Real, np.integer, np.floating)) and not isinstance(
        value,
        bool,
    ):
        return float(value)
    raise TypeError(f"Value is not float-compatible: {value!r}")


def build_router_training_df() -> pd.DataFrame:
    """Собрать синтетический reference-набор для router-модели."""
    rows: list[RouterTrainRow] = [
        (1, "M", "dwarf", 3450.0, 4.85, 0.42),
        (2, "M", "dwarf", 3520.0, 4.78, 0.45),
        (3, "M", "dwarf", 3380.0, 4.92, 0.40),
        (4, "A", "evolved", 8600.0, 3.20, 3.80),
        (5, "A", "evolved", 8450.0, 3.05, 3.60),
        (6, "A", "evolved", 8750.0, 3.30, 4.00),
    ]
    return pd.DataFrame.from_records(rows, columns=ROUTER_TRAIN_COLUMNS)


def test_router_label_roundtrip() -> None:
    """router_label должен стабильно собираться и раскладываться обратно."""
    router_label = make_router_label("m", "DWARF")

    assert router_label == "M_dwarf"
    assert split_router_label(router_label) == ("M", "dwarf")


def test_score_router_df_predicts_expected_clusters() -> None:
    """Router должен узнавать близкие кластеры на синтетическом наборе."""
    model = fit_router_model(build_router_training_df())

    rows: list[RouterScoreRow] = [
        (3490.0, 4.81, 0.43),
        (8520.0, 3.15, 3.75),
    ]
    df = pd.DataFrame.from_records(rows, columns=ROUTER_SCORE_COLUMNS)

    scored = score_router_df(model=model, df=df)

    assert list(scored["router_label"]) == ["M_dwarf", "A_evolved"]
    assert list(scored["predicted_spec_class"]) == ["M", "A"]
    assert list(scored["predicted_evolution_stage"]) == [
        "dwarf",
        "evolved",
    ]


def test_router_model_stores_effective_covariance_math() -> None:
    """В сериализованной модели должны лежать согласованные cov-поля."""
    model = fit_router_model(build_router_training_df())

    classes: Any = model["classes"]
    for params in classes.values():
        effective_cov = np.array(params["effective_cov"], dtype=float)
        inv_cov = np.array(params["inv_cov"], dtype=float)
        log_det_cov = float(params["log_det_cov"])

        assert effective_cov.shape == (3, 3)
        assert np.isfinite(log_det_cov)
        assert np.allclose(
            effective_cov @ inv_cov,
            np.eye(3),
            atol=1e-6,
        )


def test_score_router_df_adds_posterior_columns() -> None:
    """Router scoring должен возвращать posterior-диагностики."""
    model = fit_router_model(build_router_training_df())
    sample_rows: list[RouterScoreRow] = [
        (3460.0, 4.88, 0.41),
        (8710.0, 3.24, 3.90),
    ]
    sample = pd.DataFrame.from_records(
        sample_rows,
        columns=ROUTER_SCORE_COLUMNS,
    )

    scored = score_router_df(model=model, df=sample)

    assert {
        "router_log_likelihood",
        "router_log_posterior",
        "posterior_margin",
    }.issubset(scored.columns)
    assert np.isfinite(scored["router_log_likelihood"]).all()
    assert np.isfinite(scored["router_log_posterior"]).all()
    assert np.isfinite(scored["posterior_margin"]).all()


def test_router_prefers_posterior_winner_over_distance_winner() -> None:
    """Router должен выбирать posterior-winner, даже если distance меньше у другого класса."""
    model: Any = {
        "global_mu": [0.0, 0.0, 0.0],
        "global_sigma": [1.0, 1.0, 1.0],
        "features": ROUTER_SCORE_COLUMNS,
        "meta": {
            "model_version": "gaussian_router_v1",
            "source_view": "synthetic",
            "shrink_alpha": 0.15,
            "min_class_size": 3,
            "score_mode": "gaussian_log_posterior_v1",
            "prior_mode": "uniform",
        },
        "classes": {
            "A_dwarf": {
                "n": 10,
                "spec_class": "A",
                "evolution_stage": "dwarf",
                "mu": [0.0, 0.0, 0.0],
                "cov": [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                "effective_cov": [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                "inv_cov": [
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 10.0],
                ],
                "log_det_cov": float(np.log(0.001)),
            },
            "G_dwarf": {
                "n": 10,
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "mu": [1.5, 0.0, 0.0],
                "cov": [
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 10.0],
                ],
                "effective_cov": [
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 10.0],
                ],
                "inv_cov": [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                "log_det_cov": float(np.log(1000.0)),
            },
        },
    }
    sample_rows: list[RouterScoreRow] = [
        (0.9, 0.0, 0.0),
    ]
    sample = pd.DataFrame.from_records(
        sample_rows,
        columns=ROUTER_SCORE_COLUMNS,
    )

    scored = score_router_df(model=model, df=sample)

    assert str(scored.at[0, "router_label"]) == "A_dwarf"
    assert str(scored.at[0, "predicted_spec_class"]) == "A"
    assert scalar_to_float(scored.at[0, "d_mahal_router"]) > 1.0
    assert str(scored.at[0, "second_best_label"]) == "G_dwarf"
    assert scalar_to_float(scored.at[0, "margin"]) > 0.0
    assert scalar_to_float(scored.at[0, "posterior_margin"]) > 0.0


def test_router_model_save_load_roundtrip(tmp_path: Path) -> None:
    """Сохранённая router-модель должна читаться без потери поведения."""
    model = fit_router_model(build_router_training_df())
    model_path = tmp_path / "router_model.json"

    save_router_model(model, str(model_path))
    restored = load_router_model(str(model_path))

    sample_rows: list[RouterScoreRow] = [
        (3460.0, 4.88, 0.41),
    ]
    sample = pd.DataFrame.from_records(
        sample_rows,
        columns=ROUTER_SCORE_COLUMNS,
    )

    original_score = score_router_df(model=model, df=sample)
    restored_score = score_router_df(model=restored, df=sample)

    restored_meta: Any = restored["meta"]
    model_meta: Any = model["meta"]
    original_label: Any = original_score.at[0, "router_label"]
    restored_label: Any = restored_score.at[0, "router_label"]
    original_class: Any = original_score.at[0, "predicted_spec_class"]
    restored_class: Any = restored_score.at[0, "predicted_spec_class"]
    restored_params: Any = next(iter(restored["classes"].values()))

    assert str(restored_meta["model_version"]) == str(model_meta["model_version"])
    assert str(original_label) == str(restored_label)
    assert str(original_class) == str(restored_class)
    assert "effective_cov" in restored_params
    assert "log_det_cov" in restored_params
