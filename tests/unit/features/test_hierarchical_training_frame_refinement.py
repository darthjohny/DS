# Тестовый файл `test_hierarchical_training_frame_refinement.py` домена `features`.
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

from exohost.features.hierarchical_training_frame import (
    prepare_gaia_mk_refinement_training_frame,
)


def test_prepare_gaia_mk_refinement_training_frame_maps_mk_labels() -> None:
    rows: list[dict[str, object]] = []
    for source_id in range(11, 26):
        rows.append(
            {
                "source_id": source_id,
                "spectral_class": " g ",
                "spectral_subclass": " g2 ",
                "luminosity_class": " IV ",
                "peculiarity_suffix": "e",
                "teff_gspphot": "5600",
                "logg_gspphot": "4.3",
                "mh_gspphot": "0.1",
                "bp_rp": "0.74",
                "parallax": "14.0",
                "parallax_over_error": "20.0",
                "ruwe": "1.0",
                "radius_flame": "1.1",
            }
        )
    for source_id in range(26, 41):
        rows.append(
            {
                "source_id": source_id,
                "spectral_class": "k",
                "spectral_subclass": "K4",
                "luminosity_class": pd.NA,
                "peculiarity_suffix": pd.NA,
                "teff_gspphot": "4700",
                "logg_gspphot": "4.5",
                "mh_gspphot": "-0.1",
                "bp_rp": "1.05",
                "parallax": "11.0",
                "parallax_over_error": "18.0",
                "ruwe": "1.1",
                "radius_flame": "0.9",
            }
        )

    refinement_df = pd.DataFrame(rows)

    prepared = prepare_gaia_mk_refinement_training_frame(refinement_df)

    assert sorted(prepared["spec_class"].drop_duplicates().tolist()) == ["G", "K"]
    assert sorted(prepared["spec_subclass"].drop_duplicates().tolist()) == ["G2", "K4"]
    assert prepared.loc[0, "evolution_stage"] == "evolved"
    k_rows = prepared.loc[prepared["spec_subclass"] == "K4", "evolution_stage"]
    assert k_rows.notna().sum() == 0
    assert sorted(prepared["radius_gspphot"].drop_duplicates().tolist()) == [0.9, 1.1]


def test_prepare_gaia_mk_refinement_training_frame_expands_digit_only_subclass() -> None:
    refinement_df = pd.DataFrame(
        [
            {
                "source_id": source_id,
                "spectral_class": "m",
                "spectral_subclass": "4",
                "luminosity_class": "V",
                "teff_gspphot": "3200",
                "logg_gspphot": "4.9",
                "mh_gspphot": "0.0",
                "bp_rp": "2.1",
                "parallax": "18.0",
                "parallax_over_error": "25.0",
                "ruwe": "1.0",
                "radius_flame": "0.3",
            }
            for source_id in range(21, 36)
        ]
    )

    prepared = prepare_gaia_mk_refinement_training_frame(refinement_df)

    assert prepared.loc[0, "spec_subclass"] == "M4"


def test_prepare_gaia_mk_refinement_training_frame_filters_rare_subclasses() -> None:
    rows: list[dict[str, object]] = []
    for source_id in range(1, 17):
        rows.append(
            {
                "source_id": source_id,
                "spectral_class": "g",
                "spectral_subclass": "2",
                "luminosity_class": "V",
                "teff_gspphot": "5600",
                "logg_gspphot": "4.3",
                "mh_gspphot": "0.1",
                "bp_rp": "0.74",
                "parallax": "14.0",
                "parallax_over_error": "20.0",
                "ruwe": "1.0",
                "radius_flame": "1.1",
            }
        )

    rows.append(
        {
            "source_id": 99,
            "spectral_class": "o",
            "spectral_subclass": "3",
            "luminosity_class": "V",
            "teff_gspphot": "32000",
            "logg_gspphot": "4.0",
            "mh_gspphot": "0.0",
            "bp_rp": "-0.2",
            "parallax": "2.0",
            "parallax_over_error": "12.0",
            "ruwe": "1.0",
            "radius_flame": "7.0",
        }
    )

    prepared = prepare_gaia_mk_refinement_training_frame(pd.DataFrame(rows))

    assert "G2" in prepared["spec_subclass"].tolist()
    assert "O3" not in prepared["spec_subclass"].tolist()
