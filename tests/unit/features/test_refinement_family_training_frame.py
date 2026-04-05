# Тестовый файл `test_refinement_family_training_frame.py` домена `features`.
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

from exohost.features.refinement_family_training_frame import (
    prepare_gaia_mk_refinement_family_training_frame,
)


def test_prepare_gaia_mk_refinement_family_training_frame_normalizes_digits() -> None:
    family_df = pd.DataFrame(
        [
            {
                "source_id": 2,
                "spectral_class": " g ",
                "spectral_subclass": " G4 ",
                "full_subclass_label": "G4",
                "teff_gspphot": "5600",
                "logg_gspphot": "4.3",
                "mh_gspphot": "0.1",
                "bp_rp": "0.74",
                "parallax": "14.0",
                "parallax_over_error": "20.0",
                "ruwe": "1.0",
                "radius_flame": "1.1",
                "lum_flame": "1.2",
                "evolstage_flame": "524",
                "phot_g_mean_mag": "10.5",
            },
            {
                "source_id": 1,
                "spectral_class": "G",
                "spectral_subclass": 2,
                "full_subclass_label": "G2",
                "teff_gspphot": "5700",
                "logg_gspphot": "4.2",
                "mh_gspphot": "0.0",
                "bp_rp": "0.72",
                "parallax": "15.0",
                "parallax_over_error": "21.0",
                "ruwe": "1.0",
                "radius_flame": "1.0",
                "lum_flame": "1.1",
                "evolstage_flame": "523",
                "phot_g_mean_mag": "10.0",
            },
        ]
    )

    prepared = prepare_gaia_mk_refinement_family_training_frame(
        family_df,
        spectral_class="G",
    )

    assert prepared["source_id"].tolist() == [1, 2]
    assert prepared["spectral_class"].tolist() == ["G", "G"]
    assert prepared["spectral_subclass"].tolist() == ["2", "4"]
    assert prepared["full_subclass_label"].tolist() == ["G2", "G4"]
    assert prepared["evolstage_flame"].dtype.kind == "f"


def test_prepare_gaia_mk_refinement_family_training_frame_rejects_mixed_classes() -> None:
    family_df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "spectral_class": "G",
                "spectral_subclass": "2",
                "full_subclass_label": "G2",
                "teff_gspphot": "5700",
                "logg_gspphot": "4.2",
                "mh_gspphot": "0.0",
                "bp_rp": "0.72",
                "parallax": "15.0",
                "parallax_over_error": "21.0",
                "ruwe": "1.0",
                "radius_flame": "1.0",
                "lum_flame": "1.1",
                "evolstage_flame": "523",
                "phot_g_mean_mag": "10.0",
            },
            {
                "source_id": 2,
                "spectral_class": "K",
                "spectral_subclass": "4",
                "full_subclass_label": "K4",
                "teff_gspphot": "4700",
                "logg_gspphot": "4.4",
                "mh_gspphot": "-0.1",
                "bp_rp": "1.0",
                "parallax": "11.0",
                "parallax_over_error": "18.0",
                "ruwe": "1.1",
                "radius_flame": "0.8",
                "lum_flame": "0.7",
                "evolstage_flame": "520",
                "phot_g_mean_mag": "11.2",
            },
        ]
    )

    with pytest.raises(ValueError, match="unexpected spectral classes"):
        prepare_gaia_mk_refinement_family_training_frame(
            family_df,
            spectral_class="G",
        )
