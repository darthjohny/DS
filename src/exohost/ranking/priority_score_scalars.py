# Файл `priority_score_scalars.py` слоя `ranking`.
#
# Этот файл отвечает только за:
# - логики приоритизации и наблюдательной пригодности;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ranking` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import math
from numbers import Real

import pandas as pd


def is_missing_scalar(value: object) -> bool:
    # Проверяем пропуск только для скалярных значений, без неоднозначного pandas-bool.
    if value is None or value is pd.NA:
        return True

    if not isinstance(value, (Real, str)):
        return False

    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def coerce_optional_float(value: object) -> float | None:
    # Переводим необязательное скалярное значение в float или None.
    if is_missing_scalar(value):
        return None
    if not isinstance(value, (Real, str)):
        raise TypeError("Expected scalar value convertible to float.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("Expected value convertible to float.") from exc


__all__ = [
    "coerce_optional_float",
    "is_missing_scalar",
]
