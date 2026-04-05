# Тестовый файл `test_hierarchical_training_frame_ood.py` домена `features`.
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

from exohost.features.hierarchical_training_frame import (
    prepare_gaia_id_ood_training_frame,
)


def test_prepare_gaia_id_ood_training_frame_collapses_multi_membership() -> None:
    ood_df = pd.DataFrame(
        [
            {
                "source_id": 100,
                "domain_target": "ood",
                "ood_group": "ood_emission_line",
                "ood_membership_count": 2,
                "has_multi_ood_membership": True,
                "teff_gspphot": "6500",
                "logg_gspphot": "4.1",
                "mh_gspphot": "-0.2",
                "bp_rp": "0.3",
                "parallax": "4.0",
                "parallax_over_error": "8.0",
                "ruwe": "1.1",
                "selector_score_1": "0.61",
                "selector_score_2": "0.10",
            },
            {
                "source_id": 100,
                "domain_target": "OOD",
                "ood_group": "ood_white_dwarf",
                "ood_membership_count": 2,
                "has_multi_ood_membership": True,
                "teff_gspphot": "6500",
                "logg_gspphot": "4.1",
                "mh_gspphot": "-0.2",
                "bp_rp": "0.3",
                "parallax": "4.0",
                "parallax_over_error": "8.0",
                "ruwe": "1.1",
                "selector_score_1": "0.93",
                "selector_score_2": "0.42",
            },
            {
                "source_id": 101,
                "domain_target": "id",
                "ood_group": pd.NA,
                "ood_membership_count": 1,
                "has_multi_ood_membership": False,
                "teff_gspphot": "5000",
                "logg_gspphot": "4.5",
                "mh_gspphot": "0.0",
                "bp_rp": "0.9",
                "parallax": "10.0",
                "parallax_over_error": "12.0",
                "ruwe": "1.0",
                "selector_score_1": pd.NA,
                "selector_score_2": pd.NA,
            },
        ]
    )

    prepared = prepare_gaia_id_ood_training_frame(ood_df)

    assert prepared["source_id"].tolist() == [100, 101]
    assert prepared.loc[0, "domain_target"] == "ood"
    assert prepared.loc[0, "ood_group"] == "multi_ood"
    assert prepared.loc[0, "ood_group_members"] == "ood_emission_line,ood_white_dwarf"
    assert prepared.loc[0, "selector_score_1"] == pytest.approx(0.93)
    assert prepared.loc[0, "selector_score_2"] == pytest.approx(0.42)


def test_prepare_gaia_id_ood_training_frame_rejects_invalid_domain_target() -> None:
    ood_df = pd.DataFrame(
        [
            {
                "source_id": 100,
                "domain_target": "maybe",
                "teff_gspphot": "6500",
                "logg_gspphot": "4.1",
                "mh_gspphot": "-0.2",
                "bp_rp": "0.3",
                "parallax": "4.0",
                "parallax_over_error": "8.0",
                "ruwe": "1.1",
            }
        ]
    )

    with pytest.raises(ValueError, match="unsupported domain_target"):
        prepare_gaia_id_ood_training_frame(ood_df)
