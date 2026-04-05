# Тестовый файл `priority_score_testkit.py` домена `ranking`.
#
# Этот файл проверяет только:
# - проверку логики домена: priority- и observability-логики;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `ranking` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from numbers import Real

import pandas as pd


def get_float_cell(df: pd.DataFrame, row_index: int, column_name: str) -> float:
    # Читаем numeric-ячейку через явную runtime-проверку типа.
    value = df.loc[row_index, column_name]
    if isinstance(value, Real):
        return float(value)
    raise AssertionError(f"Expected numeric value in column {column_name}.")


def get_str_cell(df: pd.DataFrame, row_index: int, column_name: str) -> str:
    # Читаем строковую ячейку через явную runtime-проверку типа.
    value = df.loc[row_index, column_name]
    if isinstance(value, str):
        return value
    raise AssertionError(f"Expected string value in column {column_name}.")


def build_candidate_frame() -> pd.DataFrame:
    # Базовый synthetic набор кандидатов для ranking-логики.
    return pd.DataFrame(
        [
            {
                "source_id": "1",
                "spec_class": "G",
                "evolution_stage": "dwarf",
                "host_similarity_score": 0.91,
                "parallax": 16.0,
                "phot_g_mean_mag": 10.8,
                "parallax_over_error": 18.0,
                "ruwe": 1.02,
                "validation_factor": 0.93,
            },
            {
                "source_id": "2",
                "spec_class": "K",
                "evolution_stage": "evolved",
                "host_similarity_score": 0.64,
                "parallax": 7.0,
                "phot_g_mean_mag": 13.5,
                "parallax_over_error": 10.0,
                "ruwe": 1.18,
                "validation_factor": 0.70,
            },
            {
                "source_id": "3",
                "spec_class": "A",
                "evolution_stage": "dwarf",
                "host_similarity_score": 0.98,
                "parallax": 20.0,
                "phot_g_mean_mag": 9.5,
                "parallax_over_error": 25.0,
                "ruwe": 1.01,
                "validation_factor": 1.00,
            },
        ]
    )
