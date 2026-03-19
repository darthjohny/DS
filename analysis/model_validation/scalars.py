"""Typed helpers для приведения pandas scalar в простые Python-значения."""

from __future__ import annotations

from numbers import Integral, Real


def scalar_to_float(value: object) -> float:
    """Преобразовать pandas-скаляр в `float` с явной runtime-проверкой."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"Value is not float-compatible: {value!r}")


def scalar_to_int(value: object) -> int:
    """Преобразовать pandas-скаляр в `int` с явной runtime-проверкой."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        float_value = float(value)
        if float_value.is_integer():
            return int(float_value)
    raise TypeError(f"Value is not int-compatible: {value!r}")


__all__ = ["scalar_to_float", "scalar_to_int"]
