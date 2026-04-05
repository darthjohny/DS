# Тестовый файл `test_preprocessing.py` домена `features`.
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

from exohost.features.preprocessing import (
    NumericPreprocessingConfig,
    build_numeric_preprocessor,
)


def test_build_numeric_preprocessor_imputes_and_scales() -> None:
    # Проверяем, что preprocessing заполняет пропуски и сохраняет число признаков.
    frame = pd.DataFrame(
        {
            "teff_gspphot": [5800.0, None, 5100.0],
            "logg_gspphot": [4.4, 4.6, 4.2],
        }
    )

    preprocessor = build_numeric_preprocessor(
        ("teff_gspphot", "logg_gspphot"),
        config=NumericPreprocessingConfig(scale_numeric=True),
    )
    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape == (3, 2)


def test_build_numeric_preprocessor_can_skip_scaling() -> None:
    # Проверяем, что preprocessing может работать и без стандартизации.
    frame = pd.DataFrame(
        {
            "teff_gspphot": [5800.0, None, 5100.0],
        }
    )

    preprocessor = build_numeric_preprocessor(
        ("teff_gspphot",),
        config=NumericPreprocessingConfig(scale_numeric=False),
    )
    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape == (3, 1)
