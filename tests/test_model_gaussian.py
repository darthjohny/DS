"""Тесты для Gaussian similarity модели."""

from __future__ import annotations

import pandas as pd

from model_gaussian import (
    choose_m_subclass_label,
    fit_gaussian_model,
    score_df,
    score_one,
)

HOST_COLUMNS = [
    "spec_class",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
]

HostTrainingRow = tuple[str, float, float, float]


def build_host_training_df() -> pd.DataFrame:
    """Собрать компактный host-like train для Gaussian similarity."""
    rows: list[HostTrainingRow] = [
        ("K", 4820.0, 4.63, 0.83),
        ("K", 4760.0, 4.58, 0.79),
        ("K", 4880.0, 4.68, 0.86),
        ("G", 5670.0, 4.44, 1.04),
        ("G", 5740.0, 4.50, 1.02),
        ("G", 5600.0, 4.39, 0.98),
    ]
    return pd.DataFrame.from_records(rows, columns=HOST_COLUMNS)


def test_choose_m_subclass_label_boundaries() -> None:
    """Подклассы M должны уважать заданные температурные границы."""
    assert choose_m_subclass_label(3600.0) == "M_EARLY"
    assert choose_m_subclass_label(3300.0) == "M_MID"
    assert choose_m_subclass_label(2800.0) == "M_LATE"


def test_score_one_similarity_drops_with_distance() -> None:
    """Чем дальше объект от центра класса, тем ниже similarity."""
    model = fit_gaussian_model(
        build_host_training_df(),
        use_m_subclasses=False,
        shrink_alpha=0.10,
    )

    near_score = score_one(model, "K", 4810.0, 4.62, 0.84)
    far_score = score_one(model, "K", 6400.0, 3.70, 1.90)

    assert near_score["similarity"] > far_score["similarity"]
    assert near_score["d_mahal"] < far_score["d_mahal"]


def test_score_df_adds_expected_columns() -> None:
    """score_df должен добавлять стандартные выходные колонки."""
    model = fit_gaussian_model(
        build_host_training_df(),
        use_m_subclasses=False,
        shrink_alpha=0.10,
    )
    rows: list[HostTrainingRow] = [
        ("K", 4790.0, 4.60, 0.82),
        ("G", 5710.0, 4.46, 1.01),
    ]
    df = pd.DataFrame.from_records(rows, columns=HOST_COLUMNS)

    scored = score_df(model=model, df=df)

    assert {"gauss_label", "d_mahal", "similarity"}.issubset(scored.columns)
    assert list(scored["gauss_label"]) == ["K", "G"]
