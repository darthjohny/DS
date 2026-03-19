"""Тесты для Gaussian similarity модели."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from host_model import (
    build_contrastive_subsets,
    choose_m_subclass_label,
    fit_contrastive_gaussian_model,
    fit_gaussian_model,
    load_model,
    prepare_contrastive_training_df,
    save_model,
    score_df,
    score_df_contrastive,
    score_one,
    score_one_contrastive,
    validate_host_model_artifact,
)

HOST_COLUMNS = [
    "spec_class",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
]

HostTrainingRow = tuple[str, float, float, float]
ContrastiveTrainingRow = tuple[str, object, float, float, float]


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


def build_contrastive_training_df() -> pd.DataFrame:
    """Собрать компактный host-vs-field train для контрактных проверок."""
    rows: list[ContrastiveTrainingRow] = [
        ("M", "host", 3450.0, 4.85, 0.42),
        ("M", "field", 3520.0, 4.78, 0.45),
        ("K", True, 4820.0, 4.63, 0.83),
        ("K", False, 4760.0, 4.58, 0.79),
        ("G", 1, 5670.0, 4.44, 1.04),
        ("G", 0, 5740.0, 4.50, 1.02),
        ("F", "true", 6240.0, 4.28, 1.21),
        ("F", "false", 6120.0, 4.19, 1.15),
    ]
    return pd.DataFrame.from_records(
        rows,
        columns=["spec_class", "is_host", *HOST_COLUMNS[1:]],
    )


def build_contrastive_fit_df() -> pd.DataFrame:
    """Собрать train-набор с достаточным числом host/field строк на класс."""
    rows: list[ContrastiveTrainingRow] = [
        ("M", True, 3450.0, 4.85, 0.42),
        ("M", True, 3490.0, 4.82, 0.43),
        ("M", False, 3520.0, 4.78, 0.45),
        ("M", False, 3380.0, 4.92, 0.40),
        ("K", True, 4820.0, 4.63, 0.83),
        ("K", True, 4880.0, 4.68, 0.86),
        ("K", False, 4760.0, 4.58, 0.79),
        ("K", False, 4710.0, 4.55, 0.77),
        ("G", True, 5670.0, 4.44, 1.04),
        ("G", True, 5600.0, 4.39, 0.98),
        ("G", False, 5740.0, 4.50, 1.02),
        ("G", False, 5790.0, 4.54, 1.06),
        ("F", True, 6240.0, 4.28, 1.21),
        ("F", True, 6185.0, 4.24, 1.17),
        ("F", False, 6120.0, 4.19, 1.15),
        ("F", False, 6080.0, 4.14, 1.12),
    ]
    return pd.DataFrame.from_records(
        rows,
        columns=["spec_class", "is_host", *HOST_COLUMNS[1:]],
    )


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


def test_prepare_contrastive_training_df_normalizes_host_flags() -> None:
    """Контрастивный train contract должен нормализовать host/field флаги."""
    prepared = prepare_contrastive_training_df(build_contrastive_training_df())

    assert list(prepared.columns) == [
        "spec_class",
        "is_host",
        "teff_gspphot",
        "logg_gspphot",
        "radius_gspphot",
    ]
    assert set(prepared["is_host"].unique()) == {True, False}


def test_prepare_contrastive_training_df_requires_both_populations() -> None:
    """Каждый MKGF класс должен иметь и host, и field популяции."""
    broken = build_contrastive_training_df()
    broken = broken[~((broken["spec_class"] == "F") & (broken["is_host"] == "false"))]

    with pytest.raises(ValueError, match="F"):
        prepare_contrastive_training_df(broken)


def test_build_contrastive_subsets_returns_host_and_field_frames() -> None:
    """Контрастивный split должен собирать обе популяции по классам."""
    prepared = prepare_contrastive_training_df(build_contrastive_fit_df())
    subsets = build_contrastive_subsets(
        prepared,
        population_col="is_host",
        use_m_subclasses=False,
    )

    assert {"M", "K", "G", "F"} == set(subsets)
    assert not subsets["K"]["host"].empty
    assert not subsets["K"]["field"].empty


def test_fit_contrastive_gaussian_model_stores_host_and_field_populations(
    tmp_path: Path,
) -> None:
    """Новая contrastive-модель должна сериализовать обе популяции по классам."""
    model = fit_contrastive_gaussian_model(
        build_contrastive_fit_df(),
        population_col="is_host",
        use_m_subclasses=False,
        shrink_alpha=0.10,
        min_population_size=2,
    )
    model_path: Path = tmp_path / "contrastive_model.json"

    save_model(model, str(model_path))
    restored = load_model(str(model_path))

    assert restored["meta"]["score_mode"] == "host_vs_field_log_lr_v1"
    assert restored["meta"]["population_col"] == "is_host"
    for params in restored["classes"].values():
        assert {"host", "field"} == set(params)
        assert "effective_cov" in params["host"]
        assert "effective_cov" in params["field"]
        assert "log_det_cov" in params["host"]
        assert "log_det_cov" in params["field"]


def test_score_one_contrastive_separates_host_and_field_within_class() -> None:
    """Contrastive scorer должен предпочитать host-like объект внутри класса."""
    model = fit_contrastive_gaussian_model(
        build_contrastive_fit_df(),
        population_col="is_host",
        use_m_subclasses=False,
        shrink_alpha=0.10,
        min_population_size=2,
    )

    host_like = score_one_contrastive(model, "K", 4860.0, 4.66, 0.85)
    field_like = score_one_contrastive(model, "K", 4720.0, 4.56, 0.78)

    assert host_like["label"] == "K"
    assert field_like["label"] == "K"
    assert host_like["host_log_lr"] > field_like["host_log_lr"]
    assert host_like["host_posterior"] > field_like["host_posterior"]
    assert host_like["host_posterior"] > 0.5
    assert field_like["host_posterior"] < 0.5


def test_score_df_contrastive_adds_expected_columns() -> None:
    """Новый contrastive API должен отдавать host/field scoring колонки."""
    model = fit_contrastive_gaussian_model(
        build_contrastive_fit_df(),
        population_col="is_host",
        use_m_subclasses=False,
        shrink_alpha=0.10,
        min_population_size=2,
    )
    rows: list[HostTrainingRow] = [
        ("K", 4860.0, 4.66, 0.85),
        ("G", 5625.0, 4.41, 1.00),
    ]
    df = pd.DataFrame.from_records(rows, columns=HOST_COLUMNS)

    scored = score_df_contrastive(model=model, df=df)

    assert {
        "gauss_label",
        "host_log_likelihood",
        "field_log_likelihood",
        "host_log_lr",
        "host_posterior",
    }.issubset(scored.columns)
    assert list(scored["gauss_label"]) == ["K", "G"]


def test_legacy_score_df_rejects_contrastive_model() -> None:
    """Legacy scoring API должен явно отказывать на contrastive artifact."""
    model = fit_contrastive_gaussian_model(
        build_contrastive_fit_df(),
        population_col="is_host",
        use_m_subclasses=False,
        shrink_alpha=0.10,
        min_population_size=2,
    )
    df = pd.DataFrame.from_records(
        [("K", 4860.0, 4.66, 0.85)],
        columns=HOST_COLUMNS,
    )

    with pytest.raises(ValueError, match="score_df_contrastive"):
        score_df(model=model, df=df)


def test_validate_host_model_artifact_rejects_legacy_model() -> None:
    """Runtime host artifact validator должен отбрасывать legacy JSON contract."""
    legacy_model = fit_gaussian_model(
        build_host_training_df(),
        use_m_subclasses=False,
        shrink_alpha=0.10,
    )

    with pytest.raises(ValueError, match="legacy"):
        validate_host_model_artifact(legacy_model)
