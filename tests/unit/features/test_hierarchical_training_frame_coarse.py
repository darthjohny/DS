# Тестовый файл `test_hierarchical_training_frame_coarse.py` домена `features`.
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
    prepare_gaia_id_coarse_training_frame,
)


def test_prepare_gaia_id_coarse_training_frame_maps_bool_stage_and_sorts() -> None:
    coarse_df = pd.DataFrame(
        [
            {
                "source_id": 2,
                "ra": "11.5",
                "dec": "12.5",
                "teff_gspphot": "5770",
                "logg_gspphot": "4.40",
                "mh_gspphot": "-0.1",
                "bp_rp": "0.82",
                "parallax": "10.0",
                "parallax_over_error": "15.0",
                "ruwe": "1.02",
                "spec_class": " g ",
                "is_evolved": False,
                "radius_feature": "1.0",
            },
            {
                "source_id": 1,
                "ra": "1.5",
                "dec": "2.5",
                "teff_gspphot": "4300",
                "logg_gspphot": "4.60",
                "mh_gspphot": "0.2",
                "bp_rp": "1.12",
                "parallax": "9.0",
                "parallax_over_error": "12.0",
                "ruwe": "1.01",
                "spec_class": "K",
                "is_evolved": True,
                "radius_feature": "0.8",
            },
        ]
    )

    prepared = prepare_gaia_id_coarse_training_frame(coarse_df)

    assert prepared["source_id"].tolist() == [1, 2]
    assert prepared["spec_class"].tolist() == ["K", "G"]
    assert prepared["evolution_stage"].tolist() == ["evolved", "dwarf"]
    assert prepared["teff_gspphot"].dtype.kind == "f"
