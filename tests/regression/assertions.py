# Общие assertions для регресс-тестов.
#
# Этот файл отвечает только за:
# - компактные проверки scalar-значений и обязательных колонок;
# - стабильные DataFrame-сравнения для маленьких frozen fixtures.
#
# Следующий слой:
# - regression-тесты, которые страхуют поведение системы;
# - frozen fixtures и loader-helpers этого же слоя.

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real

import pandas as pd


def require_int_scalar(value: object) -> int:
    # Нормализуем integer-like scalar из pandas/numpy и отбрасываем bool.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def require_float_scalar(value: object) -> float:
    # Нормализуем real-like scalar из pandas/numpy и отбрасываем bool.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}.")
    return float(value)


def assert_required_columns(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str],
) -> None:
    # Проверяем, что frame содержит обязательные колонки контракта.
    missing_columns = sorted(set(required_columns).difference(df.columns))
    if missing_columns:
        missing_sql = ", ".join(missing_columns)
        raise AssertionError(f"Frame is missing required columns: {missing_sql}")


def assert_small_frame_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> None:
    # Сравниваем небольшие frozen frames без зависимости от порядка колонок.
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True),
        right.reset_index(drop=True),
        check_like=True,
    )


__all__ = [
    "assert_required_columns",
    "assert_small_frame_equal",
    "require_float_scalar",
    "require_int_scalar",
]
