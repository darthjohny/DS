"""Подготовка train DataFrame для contrastive host-модели.

Модуль отвечает за:

- нормализацию `host/field` флага;
- валидацию и очистку contrastive training frame;
- подготовку общего входного контракта для fit- и DB-layer.
"""

from __future__ import annotations

from numbers import Integral

import pandas as pd

from host_model.constants import (
    CONTRASTIVE_POPULATION_COLUMN,
    DWARF_CLASSES,
    FEATURES,
)


def normalize_host_flag(value: object) -> bool:
    """Нормализовать маркер `host/field` в строгий булев флаг."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        numeric_value = int(value)
        if numeric_value in (0, 1):
            return bool(numeric_value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "host"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "field"}:
        return False
    raise ValueError(f"Unsupported host/field flag value: {value}")


def prepare_contrastive_training_df(
    df_training: pd.DataFrame,
    population_col: str = CONTRASTIVE_POPULATION_COLUMN,
) -> pd.DataFrame:
    """Провалидировать и нормализовать training frame для contrastive-режима.

    Функция:
    - оставляет только нужные колонки;
    - фильтрует набор до `M/K/G/F dwarf`;
    - удаляет строки с пропусками в обязательных полях;
    - нормализует `host/field` колонку;
    - проверяет, что для каждого класса есть обе популяции.
    """
    required = ["spec_class", population_col, *FEATURES]
    missing = [column for column in required if column not in df_training.columns]
    if missing:
        raise ValueError(
            "Contrastive training source is missing required columns: "
            f"{', '.join(missing)}"
        )

    result = df_training.loc[:, required].copy()
    result = result[result["spec_class"].isin(DWARF_CLASSES)].copy()
    if result.empty:
        raise ValueError(
            "Contrastive training source has no MKGF dwarf rows."
        )

    result = result.dropna(subset=required).reset_index(drop=True)
    if result.empty:
        raise ValueError(
            "Contrastive training source has no complete rows after NULL filtering."
        )

    result[population_col] = [
        normalize_host_flag(value) for value in result[population_col]
    ]

    available_classes = set(result["spec_class"].astype(str))
    missing_classes = [
        spec_class for spec_class in DWARF_CLASSES if spec_class not in available_classes
    ]
    if missing_classes:
        raise ValueError(
            "Contrastive training source is missing MKGF classes: "
            f"{', '.join(missing_classes)}"
        )

    incomplete_classes: list[str] = []
    for spec_class in DWARF_CLASSES:
        subset = result[result["spec_class"] == spec_class]
        has_host = bool(subset[population_col].eq(True).any())
        has_field = bool(subset[population_col].eq(False).any())
        if not (has_host and has_field):
            incomplete_classes.append(spec_class)

    if incomplete_classes:
        raise ValueError(
            "Contrastive training source must include both host and field "
            "rows for classes: "
            f"{', '.join(incomplete_classes)}"
        )

    return result


__all__ = [
    "normalize_host_flag",
    "prepare_contrastive_training_df",
]
