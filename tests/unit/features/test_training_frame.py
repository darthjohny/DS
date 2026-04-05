# Тестовый файл `test_training_frame.py` домена `features`.
#
# Этот файл проверяет только:
# - проверку логики домена: подготовку признаков и training frame-логику;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `features` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd
import pytest

from exohost.features.training_frame import (
    prepare_host_training_frame,
    prepare_router_training_frame,
)


def build_router_df() -> pd.DataFrame:
    # Базовый synthetic router dataset для нормализации.
    return pd.DataFrame(
        [
            {
                "source_id": 2,
                "ra": "11.5",
                "dec": "12.5",
                "teff_gspphot": "5770",
                "logg_gspphot": "4.40",
                "radius_gspphot": "1.0",
                "parallax": "10.0",
                "parallax_over_error": "15.0",
                "ruwe": "1.02",
                "bp_rp": "0.82",
                "mh_gspphot": "-0.1",
                "spec_class": " g ",
                "spec_subclass": " g2 ",
                "evolution_stage": " Dwarf ",
            },
            {
                "source_id": 1,
                "ra": "1.5",
                "dec": "2.5",
                "teff_gspphot": "4300",
                "logg_gspphot": "4.60",
                "radius_gspphot": "0.8",
                "parallax": "9.0",
                "parallax_over_error": "12.0",
                "ruwe": "1.01",
                "bp_rp": "1.12",
                "mh_gspphot": "0.2",
                "spec_class": "K",
                "spec_subclass": "K4",
                "evolution_stage": "dwarf",
            },
        ]
    )


def test_prepare_router_training_frame_normalizes_labels_and_sorts() -> None:
    # Проверяем нормализацию и детерминированную сортировку по source_id.
    prepared = prepare_router_training_frame(build_router_df())

    assert prepared["source_id"].tolist() == [1, 2]
    assert prepared["spec_class"].tolist() == ["K", "G"]
    assert prepared["spec_subclass"].tolist() == ["K4", "G2"]
    assert prepared["evolution_stage"].tolist() == ["dwarf", "dwarf"]
    assert prepared["teff_gspphot"].dtype.kind == "f"


def test_prepare_router_training_frame_rejects_duplicate_source_id() -> None:
    # Дубликаты запрещаем, чтобы не допустить leakage в split.
    duplicate_df = pd.concat([build_router_df(), build_router_df().iloc[[0]]], ignore_index=True)
    duplicate_df.loc[2, "source_id"] = 1

    with pytest.raises(ValueError, match="duplicate source_id"):
        prepare_router_training_frame(duplicate_df)


def test_prepare_router_training_frame_rejects_invalid_subclass() -> None:
    # Некорректный подкласс не должен доходить до model fitting.
    invalid_df = build_router_df()
    invalid_df.loc[0, "spec_subclass"] = "G10"

    with pytest.raises(ValueError, match="unsupported spec_subclass"):
        prepare_router_training_frame(invalid_df)


def test_prepare_router_training_frame_allows_missing_subclass_values() -> None:
    # Router source может быть пригоден для coarse-задач даже без subclass-разметки.
    router_df = build_router_df()
    router_df["spec_subclass"] = pd.NA

    prepared = prepare_router_training_frame(router_df)

    assert bool(prepared["spec_subclass"].isna().all())


def test_prepare_host_training_frame_normalizes_known_host_dataset() -> None:
    # Host training frame должен проходить через ту же каноническую нормализацию.
    host_df = pd.DataFrame(
        [
            {
                "source_id": 11,
                "hostname": "Kepler-1",
                "teff_gspphot": "5600",
                "logg_gspphot": "4.3",
                "radius_flame": "1.2",
                "radius_gspphot": "1.1",
                "dist_arcsec": "0.2",
                "parallax": "14.0",
                "parallax_over_error": "20.0",
                "ruwe": "1.0",
                "phot_g_mean_mag": "11.4",
                "bp_rp": "0.74",
                "mh_gspphot": "0.1",
                "validation_factor": "0.9",
                "spec_class": "g",
                "spec_subclass": "M_mid",
                "evolution_stage": "subgiant",
            }
        ]
    )

    prepared = prepare_host_training_frame(host_df)

    assert prepared.loc[0, "spec_class"] == "G"
    assert pd.isna(prepared.loc[0, "spec_subclass"])
    assert prepared.loc[0, "evolution_stage"] == "evolved"
    assert prepared.loc[0, "radius_gspphot"] == 1.2
    assert prepared.loc[0, "radius_gspphot_legacy"] == 1.1
    assert prepared.loc[0, "phot_g_mean_mag"] == 11.4
