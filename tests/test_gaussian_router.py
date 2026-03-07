"""Тесты для Gaussian router."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gaussian_router import (
    fit_router_model,
    load_router_model,
    make_router_label,
    save_router_model,
    score_router_df,
    split_router_label,
)


def build_router_training_df() -> pd.DataFrame:
    """Собрать синтетический reference-набор для router-модели."""
    rows = [
        {
            "source_id": 1,
            "spec_class": "M",
            "evolution_stage": "dwarf",
            "teff_gspphot": 3450.0,
            "logg_gspphot": 4.85,
            "radius_gspphot": 0.42,
        },
        {
            "source_id": 2,
            "spec_class": "M",
            "evolution_stage": "dwarf",
            "teff_gspphot": 3520.0,
            "logg_gspphot": 4.78,
            "radius_gspphot": 0.45,
        },
        {
            "source_id": 3,
            "spec_class": "M",
            "evolution_stage": "dwarf",
            "teff_gspphot": 3380.0,
            "logg_gspphot": 4.92,
            "radius_gspphot": 0.40,
        },
        {
            "source_id": 4,
            "spec_class": "A",
            "evolution_stage": "evolved",
            "teff_gspphot": 8600.0,
            "logg_gspphot": 3.20,
            "radius_gspphot": 3.80,
        },
        {
            "source_id": 5,
            "spec_class": "A",
            "evolution_stage": "evolved",
            "teff_gspphot": 8450.0,
            "logg_gspphot": 3.05,
            "radius_gspphot": 3.60,
        },
        {
            "source_id": 6,
            "spec_class": "A",
            "evolution_stage": "evolved",
            "teff_gspphot": 8750.0,
            "logg_gspphot": 3.30,
            "radius_gspphot": 4.00,
        },
    ]
    return pd.DataFrame(rows)


def test_router_label_roundtrip() -> None:
    """router_label должен стабильно собираться и раскладываться обратно."""
    router_label = make_router_label("m", "DWARF")

    assert router_label == "M_dwarf"
    assert split_router_label(router_label) == ("M", "dwarf")


def test_score_router_df_predicts_expected_clusters() -> None:
    """Router должен узнавать близкие кластеры на синтетическом наборе."""
    model = fit_router_model(build_router_training_df())

    df = pd.DataFrame(
        [
            {
                "teff_gspphot": 3490.0,
                "logg_gspphot": 4.81,
                "radius_gspphot": 0.43,
            },
            {
                "teff_gspphot": 8520.0,
                "logg_gspphot": 3.15,
                "radius_gspphot": 3.75,
            },
        ]
    )

    scored = score_router_df(model=model, df=df)

    assert list(scored["router_label"]) == ["M_dwarf", "A_evolved"]
    assert list(scored["predicted_spec_class"]) == ["M", "A"]
    assert list(scored["predicted_evolution_stage"]) == [
        "dwarf",
        "evolved",
    ]


def test_router_model_save_load_roundtrip(tmp_path: Path) -> None:
    """Сохранённая router-модель должна читаться без потери поведения."""
    model = fit_router_model(build_router_training_df())
    model_path = tmp_path / "router_model.json"

    save_router_model(model, str(model_path))
    restored = load_router_model(str(model_path))

    sample = pd.DataFrame(
        [
            {
                "teff_gspphot": 3460.0,
                "logg_gspphot": 4.88,
                "radius_gspphot": 0.41,
            }
        ]
    )

    original_score = score_router_df(model=model, df=sample)
    restored_score = score_router_df(model=restored, df=sample)

    assert restored["meta"]["model_version"] == model["meta"]["model_version"]
    assert (
        original_score.loc[0, "router_label"]
        == restored_score.loc[0, "router_label"]
    )
    assert (
        original_score.loc[0, "predicted_spec_class"]
        == restored_score.loc[0, "predicted_spec_class"]
    )
