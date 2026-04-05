# Тестовый файл `test_build_host_field_training_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd
import pytest

from exohost.datasets.build_host_field_training_dataset import (
    build_host_field_training_dataset,
)


def build_host_frame() -> pd.DataFrame:
    # Небольшой synthetic host source по двум coarse-группам.
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "spec_class": "G",
                "spec_subclass": pd.NA,
                "evolution_stage": "dwarf",
                "teff_gspphot": 5700.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 1.0,
                "parallax": 10.0,
                "parallax_over_error": 20.0,
                "ruwe": 1.0,
                "bp_rp": 0.8,
                "mh_gspphot": 0.0,
            },
            {
                "source_id": 2,
                "spec_class": "K",
                "spec_subclass": pd.NA,
                "evolution_stage": "evolved",
                "teff_gspphot": 4700.0,
                "logg_gspphot": 3.2,
                "radius_gspphot": 3.0,
                "parallax": 5.0,
                "parallax_over_error": 10.0,
                "ruwe": 1.1,
                "bp_rp": 1.1,
                "mh_gspphot": -0.1,
            },
        ]
    )


def build_router_frame() -> pd.DataFrame:
    # Synthetic field pool с запасом matched-кандидатов.
    return pd.DataFrame(
        [
            {
                "source_id": 10,
                "spec_class": "G",
                "spec_subclass": "G2",
                "evolution_stage": "dwarf",
                "teff_gspphot": 5750.0,
                "logg_gspphot": 4.5,
                "radius_gspphot": 0.95,
                "parallax": 9.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.0,
                "bp_rp": 0.79,
                "mh_gspphot": 0.1,
            },
            {
                "source_id": 11,
                "spec_class": "G",
                "spec_subclass": "G3",
                "evolution_stage": "dwarf",
                "teff_gspphot": 5650.0,
                "logg_gspphot": 4.3,
                "radius_gspphot": 1.05,
                "parallax": 11.0,
                "parallax_over_error": 17.0,
                "ruwe": 1.0,
                "bp_rp": 0.83,
                "mh_gspphot": 0.0,
            },
            {
                "source_id": 12,
                "spec_class": "K",
                "spec_subclass": "K2",
                "evolution_stage": "evolved",
                "teff_gspphot": 4600.0,
                "logg_gspphot": 3.1,
                "radius_gspphot": 3.2,
                "parallax": 4.9,
                "parallax_over_error": 9.0,
                "ruwe": 1.1,
                "bp_rp": 1.15,
                "mh_gspphot": -0.2,
            },
            {
                "source_id": 13,
                "spec_class": "K",
                "spec_subclass": "K3",
                "evolution_stage": "evolved",
                "teff_gspphot": 4550.0,
                "logg_gspphot": 3.0,
                "radius_gspphot": 3.4,
                "parallax": 4.7,
                "parallax_over_error": 8.5,
                "ruwe": 1.0,
                "bp_rp": 1.18,
                "mh_gspphot": -0.1,
            },
        ]
    )


def test_build_host_field_training_dataset_returns_balanced_frame() -> None:
    # Проверяем сборку matched host-vs-field датасета 1:1.
    result = build_host_field_training_dataset(
        build_host_frame(),
        build_router_frame(),
        field_to_host_ratio=1,
        random_state=42,
    )

    assert result.shape[0] == 4
    assert sorted(result["host_label"].unique().tolist()) == ["field", "host"]
    assert result["host_label"].value_counts().to_dict() == {"field": 2, "host": 2}


def test_build_host_field_training_dataset_rejects_small_field_pool() -> None:
    # Если matched field-группы не хватает, builder должен падать явно.
    router_frame = build_router_frame().iloc[[0, 2]].reset_index(drop=True)

    with pytest.raises(ValueError, match="Field pool does not contain enough matched rows"):
        build_host_field_training_dataset(
            build_host_frame(),
            router_frame,
            field_to_host_ratio=2,
            random_state=42,
        )
